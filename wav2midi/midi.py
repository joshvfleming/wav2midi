from collections import defaultdict
from typing import List, Optional, Self
from mido import MetaMessage, Message, MidiFile, MidiTrack
from mido.midifiles.units import second2tick
import torch
from wav2midi.constants import *

CONTROL_SUSTAIN = 64


class Event:
    """
    Represents a MIDI event.
    """

    def __init__(self, message):
        self.message = message

    def __repr__(self):
        return f"Event({self.message})"

    def is_meta(self):
        return type(self.message) == MetaMessage

    def is_sustain_change(self):
        return (
            self.message.type == "control_change"
            and self.message.control == CONTROL_SUSTAIN
        )

    def is_sustain(self):
        # Half pedal or greater counts as sustain
        return self.is_sustain_change() and self.message.value >= 64

    def is_onset(self):
        return self.message.type == "note_on" and self.message.velocity > 0

    def is_offset(self):
        # Either 'note_off' or 'note_on' with velocity of 0 counts as an offset
        return self.message.type == "note_off" or (
            self.message.type == "note_on" and self.message.velocity == 0
        )

    def note_val(self):
        return self.message.note

    def velocity(self):
        return self.message.velocity

    def time(self):
        return self.message.time


class Note:
    """
    Represents a note.
    """

    def __init__(
        self, value: int, onset: float, velocity: int, offset: Optional[float] = None
    ):
        self.value = value
        self.onset = onset
        self.velocity = velocity
        self.offset = offset

    def __repr__(self):
        return (
            f"Note(value={self.value}, "
            f"onset={self.onset}, "
            f"offset={self.offset}, "
            f"velocity={self.velocity})"
        )

    def read_file(path: str) -> List[Self]:
        """
        Reads the file, parses the midi, and returns a structure containing note
        onset, offset, and velocity information.

        Args:
            path: The path to the midi file.

        Returns:
            A list of notes read from the file.
        """
        midi = MidiFile(path, clip=True)

        notes = []
        time = 0.0
        sustain_on = False
        active_notes = {}
        sustained_notes = set()

        for message in midi:
            event = Event(message)

            # Midi event times are relative, but we're more interested in absolute time
            time += event.time()

            if event.is_meta():
                continue

            if event.is_onset():
                # Note on
                note_val = event.note_val()
                note = Note(note_val, time, event.velocity())
                notes.append(note)

                # If we already had an active note for this note value, need to end
                # that one
                if note_val in active_notes:
                    active_notes[note_val].offset = time

                active_notes[note_val] = note

            elif event.is_offset():
                # Note off
                note_val = event.note_val()
                if note_val in active_notes:
                    if sustain_on:
                        sustained_notes.add(note_val)
                    else:
                        active_notes[note_val].offset = time
                        del active_notes[note_val]

            elif event.is_sustain_change():
                sustain_on = event.is_sustain()

                if not sustain_on:
                    # Sustain released, record offsets for all the sustained notes now
                    for note_val in sustained_notes:
                        active_notes[note_val].offset = time
                        del active_notes[note_val]
                    sustained_notes.clear()

        # For any remaining active notes, record end time as offset
        for note in active_notes.values():
            note.offset = time

        return notes

    def read_tensors(
        onsets: torch.Tensor, frames: torch.Tensor, velocities: torch.Tensor
    ):
        """
        Reads tensors with onset, frame, and velocity information into Notes.

        Args:
            onsets: Binary tensor that identifies note onsets by [time hop, key].
            frames: Binary tensor that identifies frames by [time hop, key].
            velocities: Note velocities by [time hop, key].

        Returns:
            A list of Notes
        """
        notes = []
        active_notes = {}
        time = 0.0
        n_hops = onsets.shape[0]

        for hop in range(n_hops):
            # Onsets
            on = onsets[hop].nonzero().flatten().tolist()
            for note_val in on:
                velocity = velocities[hop][note_val].item()

                # If note is already active, end the previous instance
                if note_val in active_notes:
                    active_notes[note_val].offset = time

                note = Note(note_val, onset=time, velocity=velocity)
                notes.append(note)
                active_notes[note_val] = note

            # Offsets
            if hop > 0:
                frame = set(frames[hop].nonzero().flatten().tolist())
                prev_frame = frames[hop - 1].nonzero().flatten().tolist()
                for note_val in prev_frame:
                    if not note_val in frame and note_val in active_notes:
                        active_notes[note_val].offset = time
                        del active_notes[note_val]

            time += HOP_LENGTH / SAMPLE_RATE

        # For any remaining active notes, record end time as offset
        for note in active_notes.values():
            note.offset = time

        return notes


def notes_to_midi(notes: List[Note], tempo: int = MIDI_DEFAULT_TEMPO) -> MidiFile:
    mid = MidiFile(ticks_per_beat=MIDI_DEFAULT_TICKS_PER_BEAT)

    metatrack = MidiTrack()
    mid.tracks.append(metatrack)

    metatrack.append(MetaMessage(type="set_tempo", tempo=tempo, time=0))
    metatrack.append(MetaMessage(type="end_of_track", time=1))

    track = MidiTrack()
    mid.tracks.append(track)

    events = []
    for note in notes:
        events.append((note.onset, "note_on", note))
        events.append((note.offset, "note_off", note))
    events.sort(key=lambda x: x[0])

    prevtime = 0.0
    for i, (time, event_type, note) in enumerate(events):
        note_val = note.value + MIN_MIDI
        ticks = second2tick(time - prevtime, MIDI_DEFAULT_TICKS_PER_BEAT, tempo)
        track.append(
            Message(type=event_type, note=note_val, velocity=note.velocity, time=ticks)
        )
        prevtime = time

    track.append(MetaMessage(type="end_of_track", time=1))

    return mid

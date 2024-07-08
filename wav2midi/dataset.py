from copy import copy
import numpy as np
import pickle
import soundfile as sf


def compute_chunk_bounds(data, chunk_size_s, sample_rate):
    chunksize = sample_rate * chunk_size_s
    bounds = []

    for i in list(range(chunksize, len(data), chunksize)):
        j = i

        # We split at zero-crossings of the signal
        while np.sign(data[j]) == np.sign(data[j - 1]):
            j += 1
        bounds.append(j)

    return bounds


def compute_chunks(audio_data, notes, chunk_size_s, sample_rate):
    bounds = compute_chunk_bounds(audio_data, chunk_size_s, sample_rate)
    prev = 0
    notes_idx = 0

    for end in bounds:
        start_time = prev / sample_rate
        end_time = (end - 1) / sample_rate
        duration = end_time - start_time

        chunk_data = audio_data[prev:end]

        # Collect all the notes that fall within this chunk window
        chunk_notes = []
        while (
            (notes_idx < len(notes))
            and (notes[notes_idx].onset >= start_time)
            and (notes[notes_idx].onset < end_time)
        ):
            new_note = copy(notes[notes_idx])

            # Adjust onset and offset times for chunk start time
            new_note.onset -= start_time
            new_note.offset -= start_time

            if new_note.offset > duration:
                new_note.offset = duration

            chunk_notes.append(new_note)
            notes_idx += 1

        yield (start_time, end_time, chunk_data, chunk_notes)
        prev = end


def write_chunk(chunk_data, chunk_notes, path):
    sf.write(f'{path}.flac', chunk_data, 16000)
    with open(f'{path}.pkl', 'wb') as f:
        pickle.dump(chunk_notes, f)

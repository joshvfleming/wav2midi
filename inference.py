import argparse
from typing import Optional
import librosa
import torch
from mido import bpm2tempo, MidiFile
from wav2midi.constants import *
from wav2midi.midi import Note, notes_to_midi
from wav2midi.util import melspectrogram


def predict(
    model: torch.Tensor, audio_data: torch.Tensor, tempo: int = MIDI_DEFAULT_TEMPO
) -> MidiFile:
    mel = melspectrogram(audio_data)

    onset_pred = frame_pred = velocity_pred = None
    with torch.no_grad():
        onset_pred, frame_pred, velocity_pred = model(mel)

    onsets = ((onset_pred > 0.5) & (frame_pred > 0.5)).squeeze(0)
    frames = (frame_pred > 0.5).squeeze(0)
    velocities = (velocity_pred.clamp(min=0, max=1) * 80 + 10).squeeze(0).round().int()

    notes = Note.read_tensors(onsets, frames, velocities)
    return notes_to_midi(notes, tempo)


def run_inference(model_path: str, inpath: str, outpath: Optional[str]):
    data, sample_rate = librosa.load(inpath, sr=SAMPLE_RATE, mono=True, dtype="float32")

    # Infer tempo
    tempo, _ = librosa.beat.beat_track(y=data, sr=sample_rate)
    tempo = bpm2tempo(int(round(tempo[0])))

    data = torch.tensor(data).unsqueeze(0).to("cuda")

    model = torch.load(model_path).to("cuda")
    midi = predict(model, data, tempo)

    if outpath is None:
        basefile = inpath.split(".")[0]
        outpath = f"{basefile}.mid"

    midi.save(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modelpath")
    parser.add_argument("inpath")
    parser.add_argument("-o", "--outpath", type=str)

    args = parser.parse_args()

    run_inference(
        model_path=args.modelpath,
        inpath=args.inpath,
        outpath=args.outpath,
    )

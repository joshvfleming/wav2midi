import argparse
from copy import copy
from typing import Dict, List, Tuple
import uuid
import numpy as np
import pickle
import os
import sys
import soundfile as sf
from midi import Note
import torch
from torch.utils.data import Dataset
import ray
from ray.experimental import tqdm_ray
from constants import *
import midi

TRAILING_CHUNK_THRESHOLD = 0.5


class MAESTRO(Dataset):
    def __init__(self, path: str, device=DEFAULT_DEVICE):
        self.path = path
        self.files = []
        self.device = device

        for file in os.listdir(path):
            if file.endswith(".pt"):
                self.files.append(file)

    def __getitem__(self, index):
        file = os.path.join(self.path, self.files[index])
        record = torch.load(file)

        data = record["data"]
        padded = torch.zeros(SAMPLE_RATE * (CHUNK_SIZE_S + 1))
        padded[: len(data)] = data

        onsets = record["onsets"].float().to(self.device)
        frames = record["frames"].float().to(self.device)
        velocities = record["velocities"].float().to(self.device)

        return padded.to(self.device), onsets, frames, velocities

    def __len__(self):
        return len(self.files)


def compute_chunk_bounds(
    data: np.ndarray, chunk_size_s: int, sample_rate: int
) -> List[int]:
    """
    Computes sample indices for where the data should be divided into chunks.

    Args:
        data:         The raw audio sample data.
        chunk_size_s: The desired chunk size in seconds.
        sample_rate:  The audio sample rate.

    Returns:
        A list of sample indices denoting the end index of the chunk
    """
    chunksize = sample_rate * chunk_size_s
    bounds = []

    for i in list(range(chunksize, len(data), chunksize)):
        j = i

        # We split at zero-crossings of the signal
        while (j + 1 < len(data)) and (np.sign(data[j]) == np.sign(data[j + 1])):
            j += 1
        bounds.append(j)

    if len(bounds) >= 2:
        if ((bounds[-1] - bounds[-2]) / chunksize) < TRAILING_CHUNK_THRESHOLD:
            # If the last chunk size is below the threshold, merge it into the previous
            # chunk
            bounds[-1] = bounds.pop()

    return bounds


def compute_chunks(
    audio_data: np.ndarray, notes: List[Note], chunk_size_s: int, sample_rate: int
) -> List[Tuple[float, float, np.ndarray, List[Note]]]:
    """
    Breaks the audio samples and notes up into chunks

    Args:
        audio_data:   The raw audio sample data.
        notes:        The notes.
        chunk_size_s: The desired chunk size in seconds.
        sample_rate:  The audio sample rate.

    Returns:
        A tuple of (start time, end time, audio data, notes)
    """
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


def generate_note_labels(
    notes: List[Note], sample_rate: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates labels from the notes.

    Args:
        notes:       The notes to tranform into labels.
        sample_rate: The audio sample rate.

    Returns:
        Tuple containing the onsets, frames, and velocities.
    """
    n_keys = MAX_MIDI - MIN_MIDI + 1
    n_hops = (CHUNK_SIZE_S * sample_rate) // HOP_LENGTH

    onsets = torch.zeros(n_hops, n_keys, dtype=torch.uint8)
    frames = torch.zeros(n_hops, n_keys, dtype=torch.uint8)
    velocities = torch.zeros(n_hops, n_keys, dtype=torch.uint8)

    for note in notes:
        onset_hop = int(note.onset * sample_rate) // HOP_LENGTH
        onset_hop = min(onset_hop, n_hops - 1)
        offset_hop = int(note.offset * sample_rate) // HOP_LENGTH
        offset_hop = min(offset_hop, n_hops - 1)

        key = note.value - MIN_MIDI
        onsets[onset_hop, key] = 1
        frames[onset_hop:offset_hop, key] = 1
        velocities[onset_hop:offset_hop, key] = note.velocity

    return onsets, frames, velocities


def record_id(record: Dict[str, torch.Tensor]) -> int:
    """
    Generates a hashed id for the record.

    Args:
        record: The record to generate an id for.

    Returns:
        int id
    """
    mask = (1 << sys.hash_info.width) - 1
    return hash(tuple(sorted(record.items()))) & mask


@ray.remote
def process_data_source(source: str, outpath: str, progress: tqdm_ray.tqdm):
    """
    Preprocesses a single input file.

    Args:
        source:  The input file path.
        outpath: The destination path.
    """
    data, sample_rate = sf.read(f"{source}.flac", dtype="float32")
    notes = midi.Note.read_file(f"{source}.midi")

    for _, _, chunk_data, chunk_notes in compute_chunks(
        data, notes, CHUNK_SIZE_S, sample_rate
    ):
        onsets, frames, velocities = generate_note_labels(chunk_notes, sample_rate)

        record = dict(
            data=torch.tensor(chunk_data),
            onsets=onsets,
            frames=frames,
            velocities=velocities,
        )

        if not os.path.exists(outpath):
            os.makedirs(outpath)

        id = record_id(record)
        outfile = os.path.join(outpath, f"{id}.pt")
        if not os.path.exists(outfile):
            torch.save(record, outfile)

    progress.update.remote(1)


def process_dataset(inpath: str, outpath: str):
    """
    Reads the raw dataset and produces a preprocessed dataset that can be read directly
    by the training pipeline.

    Args:
        inpath:    The source path containing the raw MAESTRO dataset.
        outpath:   The destination path where the preprocessed dataset will be written.
    """
    sources = set()
    for file in os.listdir(inpath):
        if file.endswith(".flac"):
            source = file.split(".")[0]
            sources.add(os.path.join(inpath, source))

    ray.init()

    progress = ray.remote(tqdm_ray.tqdm).remote(total=len(sources))
    ray.get(
        [process_data_source.remote(source, outpath, progress) for source in sources]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath")
    parser.add_argument("outpath")
    args = parser.parse_args()

    process_dataset(args.inpath, args.outpath)

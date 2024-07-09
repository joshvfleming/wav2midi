from copy import copy
from typing import List, Tuple
import numpy as np
import pickle
import soundfile as sf
from midi import Note

TRAILING_CHUNK_THRESHOLD = 0.5


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
        while np.sign(data[j]) == np.sign(data[j + 1]):
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


def write_chunk(
    chunk_data: np.ndarray, chunk_notes: List[Note], sample_rate: int, path: str
):
    """
    Writes the chunk to disk.

    Args:
        chunk_data: The audio data.
        chunk_notes: The notes.
        path: The path where files will be written.
    """
    sf.write(f"{path}.flac", chunk_data, sample_rate)
    with open(f"{path}.pkl", "wb") as f:
        pickle.dump(chunk_notes, f)

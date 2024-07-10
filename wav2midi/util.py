import torchaudio.transforms as T
from wav2midi.constants import *

_melspectrogram = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=229,
    center=True,
    pad_mode="constant",
    power=2.0,
    norm="slaney",
    mel_scale="htk",
).to(DEFAULT_DEVICE)


def melspectrogram(audio: torch.Tensor) -> torch.Tensor:
    return _melspectrogram(audio).transpose(1, 2)

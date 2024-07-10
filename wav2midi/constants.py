import torch

SAMPLE_RATE = 16000
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 229
MIN_MIDI = 21
MAX_MIDI = 108
CHUNK_SIZE_S = 20

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

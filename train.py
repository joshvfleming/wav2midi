import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from wav2midi.constants import *
from wav2midi.dataset import MaestroDataset
from wav2midi.model import Wav2Midi
from wav2midi.util import melspectrogram


def train(datapath: str, batch_size: int, device: str = DEFAULT_DEVICE):
    dataset = MaestroDataset(datapath, device=device)
    train_set, val_set = random_split(dataset, [0.8, 0.2])
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = Wav2Midi(N_MELS, N_KEYS).to(device)

    for audio, onsets, frames, velocities in loader:
        mel = melspectrogram(audio)
        onset_pred, frame_pred, velocity_pred = model(mel)

        onset_pred = onset_pred[:,:onsets.size(1),:]
        onset_loss = F.binary_cross_entropy(onset_pred, onsets)
        print(onset_loss)

        frame_pred = frame_pred[:,:frames.size(1),:]
        frame_loss = F.binary_cross_entropy(frame_pred, frames)
        print(frame_loss)

        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath")
    parser.add_argument("outpath")
    parser.add_argument("-b", "--batch-size", type=int, default=10)
    args = parser.parse_args()

    train(args.datapath, args.batch_size)

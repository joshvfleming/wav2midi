import argparse
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from wav2midi.constants import *
from wav2midi.dataset import MaestroDataset
from wav2midi.model import Wav2Midi
from wav2midi.util import melspectrogram
from tqdm import tqdm
import wandb


def train(
    datapath: str, learning_rate: float, batch_size: int, device: str = DEFAULT_DEVICE
):
    """
    Main training loop
    """
    dataset = MaestroDataset(datapath, device=device)
    train_set, val_set = random_split(dataset, [0.8, 0.2])
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = Wav2Midi(N_MELS, N_KEYS).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    wandb.init()
    wandb.watch(model, log_freq=100)

    learning_rate_step_size = 10000
    learning_rate_gamma = 0.98

    scheduler = StepLR(
        optim, step_size=learning_rate_step_size, gamma=learning_rate_gamma
    )

    for audio, onsets, frames, velocities in tqdm(loader):
        mel = melspectrogram(audio)
        onset_pred, frame_pred, velocity_pred = model(mel)

        onset_pred = onset_pred[:, : onsets.size(1), :]
        onset_loss = F.binary_cross_entropy(onset_pred, onsets)

        frame_pred = frame_pred[:, : frames.size(1), :]
        frame_loss = F.binary_cross_entropy(frame_pred, frames)

        velocity_pred = velocity_pred[:, : velocities.size(1), :]
        velocity_loss = F.mse_loss(velocity_pred, velocities)

        optim.zero_grad()
        (onset_loss + frame_loss).backward()
        velocity_loss.backward()

        optim.step()
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath")
    parser.add_argument("outpath")
    parser.add_argument("-b", "--batch-size", type=int, default=10)
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    train(args.datapath, args.learning_rate, args.batch_size)

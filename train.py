import argparse
from typing import Tuple
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from wav2midi.constants import *
from wav2midi.dataset import MaestroDataset
from wav2midi.model import Wav2Midi
from wav2midi.util import melspectrogram
from tqdm import tqdm
import wandb


def compute_loss(
    onset_pred: torch.Tensor,
    onsets: torch.Tensor,
    frame_pred: torch.Tensor,
    frames: torch.Tensor,
    velocity_pred: torch.Tensor,
    velocities: torch.Tensor,
) -> Tuple[float, float]:
    onset_pred = onset_pred[:, : onsets.size(1), :]
    onset_loss = F.binary_cross_entropy(onset_pred, onsets)

    frame_pred = frame_pred[:, : frames.size(1), :]
    frame_loss = F.binary_cross_entropy(frame_pred, frames)

    total_frame_loss = onset_loss + frame_loss

    velocity_pred = velocity_pred[:, : velocities.size(1), :]
    velocity_loss = F.mse_loss(velocity_pred, velocities)

    return total_frame_loss, velocity_loss


def train(
    datapath: str,
    learning_rate: float,
    log_freq: int,
    eval_freq: int,
    batch_size: int,
    eval_batch_size: int,
    n_epochs: int,
    clip_gradient_norm: float,
    learning_rate_step_size: int = 10000,
    learning_rate_gamma: float = 0.98,
    device: str = DEFAULT_DEVICE,
):
    """
    Main training loop
    """
    dataset = MaestroDataset(datapath, device=device)
    train_set, val_set = random_split(dataset, [0.8, 0.2])
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False)

    model = Wav2Midi(N_MELS, N_KEYS).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    run = wandb.init()
    run.watch(model, log_freq=log_freq)

    scheduler = StepLR(
        optim, step_size=learning_rate_step_size, gamma=learning_rate_gamma
    )

    i = 0

    for epoch in range(n_epochs):
        for audio, onsets, frames, velocities in tqdm(loader):
            mel = melspectrogram(audio)
            onset_pred, frame_pred, velocity_pred = model(mel)

            total_frame_loss, velocity_loss = compute_loss(
                onset_pred, onsets, frame_pred, frames, velocity_pred, velocities
            )

            if i % log_freq == 0:
                run.log(
                    {
                        "Train Onset+Frame Loss": total_frame_loss,
                        "Train Velocity Loss": velocity_loss,
                    }
                )

            optim.zero_grad()
            total_frame_loss.backward()
            velocity_loss.backward()

            optim.step()
            scheduler.step()

            clip_grad_norm_(model.parameters(), clip_gradient_norm)

            if i % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    val_total_frame_loss = 0
                    val_velocity_loss = 0
                    for val_audio, val_onsets, val_frames, val_velocities in val_loader:
                        mel = melspectrogram(val_audio)
                        onset_pred, frame_pred, velocity_pred = model(mel)

                        total_frame_loss, velocity_loss = compute_loss(
                            onset_pred,
                            val_onsets,
                            frame_pred,
                            val_frames,
                            velocity_pred,
                            val_velocities,
                        )
                        val_total_frame_loss += total_frame_loss
                        val_velocity_loss += velocity_loss

                    run.log(
                        {
                            "Eval Onset+Frame Loss": val_total_frame_loss,
                            "Eval Velocity Loss": val_velocity_loss,
                        }
                    )

                model.train()

            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath")
    parser.add_argument("outpath")
    parser.add_argument("-b", "--batch-size", type=int, default=10)
    parser.add_argument("-l", "--learning-rate", type=float, default=6e-4)
    parser.add_argument("-f", "--log-freq", type=int, default=100)
    parser.add_argument("-e", "--eval-freq", type=int, default=200)
    parser.add_argument("-n", "--epochs", type=int, default=10)
    parser.add_argument("-c", "--clip-gradient-norm", type=float, default=3.0)
    parser.add_argument("-v", "--eval-batch-size", type=int, default=50)
    args = parser.parse_args()

    train(
        datapath=args.datapath,
        learning_rate=args.learning_rate,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        n_epochs=args.epochs,
        clip_gradient_norm=args.clip_gradient_norm,
    )

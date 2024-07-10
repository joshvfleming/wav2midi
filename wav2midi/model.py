from torch import nn
from wav2midi.constants import *
from wav2midi.util import melspectrogram


class ConvStack(nn.Module):
    def __init__(self, input_dim, output_dim, cnn_dropout=0.25, fc_dropout=0.5):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_dim // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_dim // 16, output_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_dim // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(cnn_dropout),
            nn.Conv2d(output_dim // 16, output_dim // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_dim // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(cnn_dropout),
        )

        self.fc = nn.Sequential(
            nn.Linear((output_dim // 8) * (input_dim // 4), output_dim),
            nn.Dropout(fc_dropout),
        )

    def forward(self, mel):
        # Add channel
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)

        return x


# LSTM() returns tuple of (tensor, (recurrent state))
class ExtractLSTMTensor(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor


class Wav2Midi(nn.Module):
    def __init__(self, input_dim, output_dim, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16

        self.onset_stack = nn.Sequential(
            ConvStack(input_dim, model_size),
            nn.LSTM(model_size, model_size // 2, batch_first=True, bidirectional=True),
            ExtractLSTMTensor(),
            nn.Linear(model_size, output_dim),
            nn.Sigmoid(),
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_dim, model_size),
            nn.Linear(model_size, output_dim),
            nn.Sigmoid(),
        )
        self.combined_stack = nn.Sequential(
            nn.LSTM(
                output_dim * 2, model_size // 2, batch_first=True, bidirectional=True
            ),
            ExtractLSTMTensor(),
            nn.Linear(model_size, output_dim),
            nn.Sigmoid(),
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_dim, model_size),
            nn.Linear(model_size, output_dim),
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        frame_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred, frame_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch):
        audio = batch["data"]
        onset_label = batch["onsets"]
        frame_label = batch["frames"]
        velocity_label = batch["velocities"]

        mel = melspectrogram(audio)
        onset_pred, frame_pred, velocity_pred = self(mel)

        predictions = {
            "onsets": onset_pred.reshape(*onset_label.shape),
            "frames": frame_pred.reshape(*frame_label.shape),
            "velocities": velocity_pred.reshape(*velocity_label.shape),
        }

        losses = {
            "onsets": F.binary_cross_entropy(predictions["onsets"], onset_label),
            "frames": F.binary_cross_entropy(predictions["frames"], frame_label),
            "velocities": self.velocity_loss(
                predictions["velocities"], velocity_label, onset_label
            ),
        }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (
                onset_label * (velocity_label - velocity_pred) ** 2
            ).sum() / denominator

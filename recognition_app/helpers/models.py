"""Module with models implementations."""
import lightning as L
import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import ctc_loss, log_softmax

from .utils import get_optimizer, get_scheduler


class FeatureExtractor(nn.Module):
    """Feature extractor model implementation."""

    def __init__(self, input_size: tuple[int, int] = (64, 320), output_len: int = 20):
        """FeatureExtractor instance.

        Args:
            - input_size: input image size
            - output_len: number of output features
        """
        super().__init__()

        height, width = input_size
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        self.pool = nn.AvgPool2d(kernel_size=(height // 32, 1))
        self.proj = nn.Conv2d(width // 32, output_len, kernel_size=1)
        self.num_output_features = self.cnn[-1][-1].bn2.num_features

    def apply_projection(self, input: torch.Tensor) -> torch.Tensor:
        """Use convolution to increase width of a features.

        Args:
            - input: Tensor of features (shaped B x C x H x W).

        Returns:
            New tensor of features (shaped B x C x H x W').
        """
        input = input.permute(0, 3, 2, 1).contiguous()
        input = self.proj(input)
        input = input.permute(0, 2, 3, 1).contiguous()
        return input

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features = self.cnn(input)
        features = self.pool(features)
        features = self.apply_projection(features)
        return features


class SequencePredictor(nn.Module):
    """Sequence predictor model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """SequencePredictor instance.

        Args:
            - input_size: input image size
            - hidden_size: size of hidden layer
            - num_layers: number of lstm layers
            - num_classes: number of classes to predict
            - dropout: dropout coef
            - bidirectional: is lstm bidirectional
        """
        super().__init__()

        self.num_classes = num_classes
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = nn.Linear(in_features=fc_in, out_features=num_classes)

    def _init_hidden(self, batch_size: int) -> tuple[torch.Tensor]:
        """Initialize new tensor of zeroes for RNN hidden state.

        Args:
            - batch_size: size of batch

        Returns:
            Tensors of zeros shaped (num_layers * num_directions, batch, hidden_size).
        """
        num_directions = 2 if self.rnn.bidirectional else 1

        h = torch.zeros(
            self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size
        )
        c = torch.zeros(
            self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size
        )

        return h, c

    def _reshape_features(self, input: torch.Tensor) -> torch.Tensor:
        """Change dimensions of input to fit RNN expected input.

        Args:
            - input: input shaped (B x (C=1) x H x W).

        Returns:
            New tensor shaped (W x B x H).
        """
        input = input.squeeze(1)
        input = input.permute(2, 0, 1)

        return input

    def forward(self, input: torch.Tensor, device: torch.device = "cpu") -> torch.Tensor:
        """SequencePredictor forward pass.

        Args:
            - input: input data
        """
        input = self._reshape_features(input)

        batch_size = input.size(1)
        h_0, c_0 = self._init_hidden(batch_size)
        output, _ = self.rnn(input, (h_0, c_0))

        output = self.fc(output)
        return output


class CRNN(nn.Module):
    """CRNN model implementation."""

    def __init__(
        self,
        alphabet: str,
        cnn_input_size: tuple[int] = (64, 320),
        cnn_output_len: int = 20,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.3,
        rnn_bidirectional: bool = True,
    ):
        """CRNN instance.

        Args:
            - alphabet: allowed alphabet
            - cnn_input_size: input image size
            - cnn_output_len: number of output features
            - rnn_hidden_size: size of rnn hidden layer
            - rnn_num_layers: number of lstm layers
            - rnn_dropout: dropout coef
            - rnn_bidirectional: is lstm bidirectional
        """
        super().__init__()
        self.alphabet = alphabet
        self.features_extractor = FeatureExtractor(
            input_size=cnn_input_size, output_len=cnn_output_len
        )
        self.sequence_predictor = SequencePredictor(
            input_size=self.features_extractor.num_output_features,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            num_classes=len(alphabet) + 1,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """CRNN forward pass.

        Args:
            - input: input data
        """
        features = self.features_extractor(input)
        sequence = self.sequence_predictor(features)
        return sequence


class LightningCRNN(L.LightningModule):
    """CRNN model implementation with Lightning."""

    def __init__(
        self,
        alphabet: str,
        cnn_input_size: tuple[int] = (64, 320),
        cnn_output_len: int = 20,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.3,
        rnn_bidirectional: bool = True,
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
    ):
        """Lightning CRNN instance.

        Args:
            - alphabet: allowed alphabet
            - cnn_input_size: input image size
            - cnn_output_len: number of output features
            - rnn_hidden_size: size of rnn hidden layer
            - rnn_num_layers: number of lstm layers
            - rnn_dropout: dropout coef
            - rnn_bidirectional: is lstm bidirectional
            - optimizer_config: custom config for optimizer
            - scheduler_config: custom config for scheduler
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = CRNN(
            alphabet=alphabet,
            cnn_input_size=cnn_input_size,
            cnn_output_len=cnn_output_len,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

    def forward(self, input: torch.Tensor):
        """Forward pass of the model."""
        return self.model(input)

    def basic_step(self, batch, stage: str):
        """Basic step for train and validation."""
        images = batch["image"]
        seqs_gt = batch["seq"]
        seq_lens_gt = batch["seq_len"]

        seqs_pred = self(images).cpu()
        log_probs = log_softmax(seqs_pred, dim=2)
        seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

        loss = ctc_loss(
            log_probs=log_probs,
            targets=seqs_gt,
            input_lengths=seq_lens_pred,
            target_lengths=seq_lens_gt,
        )
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        """Training step implementation."""
        loss = self.basic_step(batch, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Validation step implementation."""
        loss = self.basic_step(batch, "validation")
        return {"val_loss": loss}

    def configure_optimizers(self):
        """Configure custom optimizers."""

        optimizer = get_optimizer(self.optimizer_config, self.model.parameters())
        scheduler = get_scheduler(self.scheduler_config, optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

"""Linear classifier model for laughter event prediction."""

import torch
import torch.nn as nn


class LaughterPredictor(nn.Module):
    """Simple linear classifier for binary laughter event prediction.

    Takes 4096-dimensional transformer output and predicts binary label
    indicating whether a laughter event will occur in the prediction interval.
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = None):
        """Initialize the laughter predictor.

        Args:
            input_dim: Dimension of input features (default: 4096)
            hidden_dim: Optional hidden dimension for MLP variant.
                       If None, uses simple linear classifier.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if hidden_dim is None:
            # Simple linear classifier (recommended)
            self.classifier = nn.Linear(input_dim, 1)
        else:
            # Optional MLP variant for experimentation
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Logits [batch_size, 1] (pass through sigmoid for probabilities)
        """
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probabilities.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Probabilities [batch_size, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary labels.

        Args:
            x: Input features [batch_size, input_dim]
            threshold: Decision threshold (default: 0.5)

        Returns:
            Binary predictions [batch_size, 1]
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).float()

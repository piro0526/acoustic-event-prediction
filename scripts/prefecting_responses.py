import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
from moshi.models import loaders
from moshi.models.lm import LMGen
import sentencepiece

# Add src to Python path for LaughterPredictor
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from laughter_prediction.model import LaughterPredictor


def decode_tokens(text_tokenizer, tokens):
    pieces = []
    for token in tokens:
        try:
            piece = text_tokenizer.id_to_piece(token)
            if piece == '<unk>' or piece == '<pad>':
                piece = '*'
            pieces.append(piece)
        except Exception:
            pieces.append(f"<{token}>")
    return "".join(pieces).replace("\u2581", " ").strip()


def load_laughter_head(checkpoint_path: str, device: str, hidden_dim: int = None):
    """Load trained LaughterPredictor from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        hidden_dim: Hidden dimension if using MLP variant (None for linear)

    Returns:
        LaughterPredictor model in eval mode
    """
    model = LaughterPredictor(input_dim=4096, hidden_dim=hidden_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    metrics = checkpoint.get('metrics', {})
    print(f"Loaded laughter predictor from epoch {epoch}")
    if metrics:
        f1 = metrics.get('f1', 'N/A')
        auc = metrics.get('auc', 'N/A')
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
        auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else str(auc)
        print(f"  Best F1: {f1_str}, AUC: {auc_str}")

    return model


def add_laughter_head_to_model(moshi_lm: nn.Module, laughter_predictor: LaughterPredictor):
    """Add trained laughter prediction head to Moshi LM's extra_heads.

    This integrates the trained linear classifier as an extra head in the Moshi model,
    allowing laughter prediction during streaming inference.

    Args:
        moshi_lm: The Moshi LM model
        laughter_predictor: Trained LaughterPredictor model

    Returns:
        Index of the added laughter head in extra_heads
    """
    # Get the classifier from LaughterPredictor
    if isinstance(laughter_predictor.classifier, nn.Linear):
        # Simple linear classifier - copy weights directly
        laughter_linear = nn.Linear(
            laughter_predictor.input_dim,
            1,
            bias=laughter_predictor.classifier.bias is not None
        )
        laughter_linear.weight.data = laughter_predictor.classifier.weight.data.clone()
        if laughter_predictor.classifier.bias is not None:
            laughter_linear.bias.data = laughter_predictor.classifier.bias.data.clone()
    else:
        # MLP variant - we need to keep the full classifier
        # For simplicity, we'll wrap it
        laughter_linear = laughter_predictor.classifier

    # Move to same device and dtype as moshi_lm
    device = next(moshi_lm.parameters()).device
    dtype = next(moshi_lm.parameters()).dtype
    laughter_linear = laughter_linear.to(device=device, dtype=dtype)

    # Add to extra_heads
    laughter_head_idx = len(moshi_lm.extra_heads)
    moshi_lm.extra_heads.append(laughter_linear)

    print(f"Added laughter head to moshi_lm.extra_heads at index {laughter_head_idx}")

    return laughter_head_idx


class LMGenWithLaughter(LMGen):
    """Extended LMGen that supports laughter prediction via extra_heads."""

    def __init__(self, lm_model, laughter_head_idx: int = None, laughter_threshold: float = 0.5, **kwargs):
        super().__init__(lm_model, **kwargs)
        self.laughter_head_idx = laughter_head_idx
        self.laughter_threshold = laughter_threshold

    @torch.no_grad()
    def step_with_laughter(
        self,
        input_tokens: torch.Tensor,
        depformer_replace_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Step with laughter prediction.

        Returns:
            Tuple of (tokens_out, laughter_probs, transformer_out)
            - tokens_out: Generated tokens or None
            - laughter_probs: Laughter probabilities [B, 1] or None
            - transformer_out: Raw transformer output [B, 1, dim] or None
        """
        out = self._step(input_tokens, depformer_replace_tokens)
        if out is None:
            return None, None, None

        tokens_out, transformer_out = out

        # Compute laughter prediction using the added extra_head
        laughter_probs = None
        if self.laughter_head_idx is not None and self.laughter_head_idx < len(self.lm_model.extra_heads):
            laughter_head = self.lm_model.extra_heads[self.laughter_head_idx]
            # Apply sigmoid (not softmax) for binary classification
            logits = laughter_head(transformer_out)  # [B, 1, 1]
            laughter_probs = torch.sigmoid(logits).squeeze(-1)  # [B, 1]

        return tokens_out, laughter_probs, transformer_out


def main():
    parser = argparse.ArgumentParser(
        description='Compute features using multi-speaker processing with laughter prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help='Top-p sampling value'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--temp_text',
        type=float,
        default=0.9,
        help='Temperature for text generation'
    )
    parser.add_argument(
        '--temp_audio',
        type=float,
        default=0.9,
        help='Temperature for audio generation'
    )
    parser.add_argument(
        '--user_audio_path',
        type=str,
        required=True,
        help='Path to the user audio file'
    )
    parser.add_argument(
        '--prefetching_range_seconds',
        type=int,
        default=4,
        help='Number of seconds to prefetch audio'
    )

    # Laughter prediction arguments
    parser.add_argument(
        '--laughter_checkpoint',
        type=str,
        default=None,
        help='Path to trained LaughterPredictor checkpoint (optional)'
    )
    parser.add_argument(
        '--laughter_threshold',
        type=float,
        default=0.5,
        help='Decision threshold for laughter prediction'
    )
    parser.add_argument(
        '--laughter_hidden_dim',
        type=int,
        default=None,
        help='Hidden dimension if using MLP classifier (None for linear)'
    )

    args = parser.parse_args()

    print("Loading Moshi models...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    hf_repo = "kyutai/moshiko-pytorch-bf16"

    print("Loading Mimi...")
    mimi_path = loaders.hf_get(loaders.MIMI_NAME, hf_repo=hf_repo)
    mimi = loaders.get_mimi(mimi_path, device=device)
    mimi.set_num_codebooks(8)
    mimi.cuda()

    print("Loading Moshi LM...")
    moshi_path = loaders.hf_get(loaders.MOSHI_NAME, hf_repo=hf_repo)
    moshi_lm = loaders.get_moshi_lm(moshi_path, device=device)

    print("Loading text tokenizer...")
    tokenizer_path = loaders.hf_get(loaders.TEXT_TOKENIZER_NAME, hf_repo=hf_repo)
    text_tokenizer = sentencepiece.SentencePieceProcessor()
    text_tokenizer.load(str(tokenizer_path))

    # Load and add laughter prediction head if checkpoint provided
    laughter_head_idx = None
    if args.laughter_checkpoint:
        print(f"\nLoading laughter predictor from {args.laughter_checkpoint}...")
        laughter_predictor = load_laughter_head(
            args.laughter_checkpoint,
            device,
            hidden_dim=args.laughter_hidden_dim
        )
        laughter_head_idx = add_laughter_head_to_model(moshi_lm, laughter_predictor)

    waveform, sample_rate = torchaudio.load(args.user_audio_path)

    if sample_rate != 24000:
        print(f"Resampling from {sample_rate} Hz to 24000 Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
        waveform = resampler(waveform)

    # Add silence for prefetching_range_seconds
    silence_samples = int(args.prefetching_range_seconds * 24000)
    silence = torch.zeros((waveform.shape[0], silence_samples))

    waveform = waveform.unsqueeze(0).to(device)
    silence = silence.unsqueeze(0).to(device)

    print("\nEncoding audio with Mimi...")
    with torch.no_grad():
        input_codes = mimi.encode(waveform)  # [B, K, T]
        silence_codes = mimi.encode(silence)

    # Expand to [num_samples, K, T]
    input_codes = input_codes.repeat(args.num_samples, 1, 1)
    silence_codes = silence_codes.repeat(args.num_samples, 1, 1)

    # Use extended LMGen with laughter support
    lm_gen = LMGenWithLaughter(
        moshi_lm,
        laughter_head_idx=laughter_head_idx,
        laughter_threshold=args.laughter_threshold,
        temp=0.8,
        temp_text=0.7
    )

    response_text_tokens = [[] for _ in range(args.num_samples)]
    laughter_predictions = [[] for _ in range(args.num_samples)]  # Store per-sample predictions

    print(f'input_codes shape: {input_codes.shape}')
    print(f'silence_codes shape: {silence_codes.shape}')

    with torch.no_grad(), lm_gen.streaming(batch_size=args.num_samples):
        # Process input audio
        for step in range(input_codes.shape[-1]):
            if laughter_head_idx is not None:
                tokens_out, laughter_probs, _ = lm_gen.step_with_laughter(input_codes[:, :, step:step+1])
                if laughter_probs is not None:
                    for b in range(args.num_samples):
                        laughter_predictions[b].append(laughter_probs[b, 0].item())
            else:
                tokens_out = lm_gen.step(input_codes[:, :, step:step+1])

        # Process silence (prefetching phase)
        for step in range(silence_codes.shape[-1]):
            if laughter_head_idx is not None:
                tokens_out, laughter_probs, _ = lm_gen.step_with_laughter(silence_codes[:, :, step:step+1])
                if laughter_probs is not None:
                    for b in range(args.num_samples):
                        laughter_predictions[b].append(laughter_probs[b, 0].item())
            else:
                tokens_out = lm_gen.step(silence_codes[:, :, step:step+1])

            if tokens_out is not None:
                for b in range(args.num_samples):
                    text_token = tokens_out[b, 0, 0].item()
                    response_text_tokens[b].append(text_token)

    print("\nResponse generation:")
    for b in range(args.num_samples):
        text = decode_tokens(text_tokenizer, response_text_tokens[b])
        print(f"  Sample {b}: '{text}'")

    # Print laughter prediction summary if enabled
    if laughter_head_idx is not None:
        print("\nLaughter prediction summary:")
        for b in range(args.num_samples):
            preds = laughter_predictions[b]
            if preds:
                avg_prob = sum(preds) / len(preds)
                num_laughter = sum(1 for p in preds if p >= args.laughter_threshold)
                print(f"  Sample {b}: {num_laughter}/{len(preds)} frames predicted as laughter "
                      f"(avg prob: {avg_prob:.4f})")


if __name__ == '__main__':
    main()

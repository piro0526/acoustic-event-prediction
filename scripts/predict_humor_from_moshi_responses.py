#!/usr/bin/env python
"""Generate multiple Moshi responses and predict humor using UR-Funny2 BiLSTM model.

This script:
1. Loads user audio and encodes it with Mimi
2. Runs Moshi inference in batch to sample multiple responses
3. Collects transformer_out at each timestep during generation
4. Applies the trained BiLSTM + Attention humor predictor to each response's
   transformer_out sequence to produce a humor score

Usage:
    python scripts/predict_humor_from_moshi_responses.py \
        --user_audio_path path/to/audio.wav \
        --humor_checkpoint output/urfunny2/bilstm_prediction/best_model.pt \
        --num_samples 8 \
        --prefetch_seconds 4

    # With custom thresholds and temperatures
    python scripts/predict_humor_from_moshi_responses.py \
        --user_audio_path path/to/audio.wav \
        --humor_checkpoint output/urfunny2/bilstm_prediction/best_model.pt \
        --num_samples 16 \
        --temp_text 0.9 \
        --temp_audio 0.8 \
        --humor_threshold 0.5 \
        --output_dir output/humor_inference
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece
import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from moshi.models import loaders
from moshi.models.lm import LMGen


# ---------- BiLSTM Model (must match training definition) ----------


class BiLSTMAttentionClassifier(nn.Module):
    """BiLSTM + Attention Pooling + Linear classifier.

    [B, T, 4096] -> BiLSTM -> [B, T, hidden*2] -> attention pool -> [B, hidden*2] -> Linear -> [B, 1]
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_w = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        if mask is not None:
            lengths = mask.sum(dim=1).cpu().clamp(min=1)
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=T)
        else:
            lstm_out, _ = self.lstm(x)
        scores = self.attn_w(lstm_out).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# ---------- Utilities ----------


def decode_tokens(text_tokenizer: sentencepiece.SentencePieceProcessor, tokens: list[int]) -> str:
    pieces = []
    for token in tokens:
        try:
            piece = text_tokenizer.id_to_piece(token)
            if piece in ("<unk>", "<pad>"):
                piece = "*"
            pieces.append(piece)
        except Exception:
            pieces.append(f"<{token}>")
    return "".join(pieces).replace("\u2581", " ").strip()


def load_humor_model(checkpoint_path: str, device: torch.device) -> BiLSTMAttentionClassifier:
    """Load trained BiLSTM humor predictor from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    train_args = checkpoint.get("args", {})

    model = BiLSTMAttentionClassifier(
        input_dim=train_args.get("input_dim", 4096),
        hidden_dim=train_args.get("hidden_dim", 128),
        num_layers=train_args.get("num_layers", 1),
        dropout=0.0,  # no dropout at inference
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    dev_metrics = checkpoint.get("dev_metrics", {})
    print(f"Loaded humor predictor from epoch {epoch}")
    if dev_metrics:
        print(f"  Dev F1={dev_metrics.get('f1', 0):.4f}, AUC={dev_metrics.get('auc', 0):.4f}")

    return model


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser(
        description="Generate Moshi responses and predict humor with BiLSTM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--user_audio_path", type=str, required=True, help="Path to user audio file (.wav)")
    parser.add_argument(
        "--humor_checkpoint",
        type=str,
        default="output/urfunny2/bilstm_prediction/best_model.pt",
        help="Path to trained BiLSTM humor model checkpoint",
    )
    parser.add_argument("--num_samples", type=int, default=8, help="Number of response samples to generate")
    parser.add_argument("--prefetch_seconds", type=float, default=4.0, help="Seconds of silence after audio for response generation")
    parser.add_argument("--temp_text", type=float, default=0.7, help="Temperature for text token sampling")
    parser.add_argument("--temp_audio", type=float, default=0.8, help="Temperature for audio token sampling")
    parser.add_argument("--top_k_text", type=int, default=25, help="Top-k for text sampling")
    parser.add_argument("--top_k_audio", type=int, default=250, help="Top-k for audio sampling")
    parser.add_argument("--humor_threshold", type=float, default=0.5, help="Threshold for humor prediction")
    parser.add_argument("--hf_repo", type=str, default="kyutai/moshiko-pytorch-bf16", help="HuggingFace repo for Moshi model")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results (optional)")
    parser.add_argument("--save_audio", action="store_true", help="Save generated audio responses")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    # --- Load models ---
    print("\nLoading Mimi...")
    mimi_path = loaders.hf_get(loaders.MIMI_NAME, hf_repo=args.hf_repo)
    mimi = loaders.get_mimi(mimi_path, device=str(device))
    mimi.set_num_codebooks(8)

    print("Loading Moshi LM...")
    moshi_path = loaders.hf_get(loaders.MOSHI_NAME, hf_repo=args.hf_repo)
    moshi_lm = loaders.get_moshi_lm(moshi_path, device=str(device))

    print("Loading text tokenizer...")
    tokenizer_path = loaders.hf_get(loaders.TEXT_TOKENIZER_NAME, hf_repo=args.hf_repo)
    text_tokenizer = sentencepiece.SentencePieceProcessor()
    text_tokenizer.load(str(tokenizer_path))

    print("Loading humor predictor...")
    humor_model = load_humor_model(args.humor_checkpoint, device)

    # --- Load and prepare audio ---
    print(f"\nLoading audio: {args.user_audio_path}")
    waveform, sample_rate = torchaudio.load(args.user_audio_path)

    if sample_rate != mimi.sample_rate:
        print(f"Resampling from {sample_rate} Hz to {mimi.sample_rate} Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=mimi.sample_rate)
        waveform = resampler(waveform)

    # Mono, [1, 1, samples]
    if waveform.shape[0] > 1:
        waveform = waveform[:1]
    waveform = waveform.unsqueeze(0).to(device)

    # Silence for response generation window
    silence_samples = int(args.prefetch_seconds * mimi.sample_rate)
    silence = torch.zeros((1, 1, silence_samples), device=device)

    # --- Encode audio ---
    print("Encoding audio with Mimi...")
    with torch.no_grad():
        input_codes = mimi.encode(waveform)  # [1, K, T_input]
        silence_codes = mimi.encode(silence)  # [1, K, T_silence]

    T_input = input_codes.shape[-1]
    T_silence = silence_codes.shape[-1]
    T_total = T_input + T_silence

    # Expand for batch sampling
    input_codes = input_codes.expand(args.num_samples, -1, -1)
    silence_codes = silence_codes.expand(args.num_samples, -1, -1)

    print(f"  Input frames: {T_input}, Silence frames: {T_silence}, Total: {T_total}")
    print(f"  Generating {args.num_samples} response samples...\n")

    # --- Streaming inference ---
    lm_gen = LMGen(
        moshi_lm,
        use_sampling=True,
        temp=args.temp_audio,
        temp_text=args.temp_text,
        top_k=args.top_k_audio,
        top_k_text=args.top_k_text,
    )

    transformer_outs = [[] for _ in range(args.num_samples)]  # per-sample transformer_out
    response_text_tokens = [[] for _ in range(args.num_samples)]
    response_audio_tokens = [[] for _ in range(args.num_samples)]

    start_time = time.time()

    with torch.no_grad(), lm_gen.streaming(batch_size=args.num_samples):
        # Process input audio
        for step in range(T_input):
            result = lm_gen._step(input_codes[:, :, step : step + 1])
            if result is not None:
                tokens_out, transformer_out = result
                # transformer_out: [B, 1, dim] or [2B, 1, dim] with CFG
                tout = transformer_out[:args.num_samples]  # take first B in case of CFG
                for b in range(args.num_samples):
                    transformer_outs[b].append(tout[b, 0].float().cpu())

        # Process silence (response generation window)
        for step in range(T_silence):
            result = lm_gen._step(silence_codes[:, :, step : step + 1])
            if result is not None:
                tokens_out, transformer_out = result
                tout = transformer_out[:args.num_samples]
                for b in range(args.num_samples):
                    transformer_outs[b].append(tout[b, 0].float().cpu())
                    text_token = tokens_out[b, 0, 0].item()
                    response_text_tokens[b].append(text_token)
                    if tokens_out.shape[1] > 1:
                        response_audio_tokens[b].append(tokens_out[b, 1:, 0].cpu())

    gen_time = time.time() - start_time
    print(f"Generation completed in {gen_time:.1f}s ({T_total / gen_time:.1f} steps/s)\n")

    # --- Humor prediction on collected transformer_out sequences ---
    print("Running humor prediction on transformer_out sequences...")

    # Stack per-sample sequences: each is [T_i, dim]
    features_list = [torch.stack(tos) for tos in transformer_outs]  # list of [T_i, 4096]
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[1]

    # Pad and create mask
    padded = torch.zeros(args.num_samples, max_len, dim)
    mask = torch.zeros(args.num_samples, max_len, dtype=torch.bool)
    for i, (feat, length) in enumerate(zip(features_list, lengths)):
        padded[i, :length] = feat
        mask[i, :length] = True

    padded = padded.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        logits = humor_model(padded, mask)  # [B, 1]
        humor_probs = torch.sigmoid(logits).squeeze(-1)  # [B]

    # --- Display results ---
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    results = []
    for b in range(args.num_samples):
        text = decode_tokens(text_tokenizer, response_text_tokens[b])
        prob = humor_probs[b].item()
        is_funny = prob >= args.humor_threshold
        label = "FUNNY" if is_funny else "NOT FUNNY"

        result = {
            "sample_id": b,
            "text": text,
            "humor_prob": prob,
            "is_funny": is_funny,
            "num_frames": lengths[b],
        }
        results.append(result)

        marker = "*" if is_funny else " "
        print(f"\n[{marker}] Sample {b} (humor={prob:.4f}, frames={lengths[b]}): {label}")
        print(f"    Text: '{text}'")

    # Summary
    num_funny = sum(1 for r in results if r["is_funny"])
    avg_prob = sum(r["humor_prob"] for r in results) / len(results)
    print(f"\n{'=' * 70}")
    print(f"Summary: {num_funny}/{args.num_samples} samples predicted as funny")
    print(f"  Average humor probability: {avg_prob:.4f}")
    print(f"  Threshold: {args.humor_threshold}")

    # Sort by humor probability
    sorted_results = sorted(results, key=lambda r: r["humor_prob"], reverse=True)
    print(f"\n  Top 3 funniest:")
    for r in sorted_results[:3]:
        print(f"    Sample {r['sample_id']}: prob={r['humor_prob']:.4f} - '{r['text'][:60]}'")

    # --- Save results ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        predictions = {
            "user_audio": args.user_audio_path,
            "num_samples": args.num_samples,
            "humor_threshold": args.humor_threshold,
            "temp_text": args.temp_text,
            "temp_audio": args.temp_audio,
            "generation_time_seconds": gen_time,
            "summary": {
                "num_funny": num_funny,
                "avg_humor_prob": avg_prob,
            },
            "results": results,
        }
        pred_path = output_dir / "humor_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"\nPredictions saved to {pred_path}")

        # Save transformer_out features (optional, for analysis)
        for b in range(args.num_samples):
            feat = features_list[b].numpy()
            np.save(output_dir / f"transformer_out_sample_{b}.npy", feat)

        # Save audio if requested
        if args.save_audio and response_audio_tokens[0]:
            print("Decoding and saving audio responses...")
            for b in range(args.num_samples):
                if response_audio_tokens[b]:
                    audio_codes = torch.stack(response_audio_tokens[b], dim=-1)  # [dep_q, T]
                    audio_codes = audio_codes.unsqueeze(0).to(device)  # [1, dep_q, T]
                    with torch.no_grad():
                        pcm = mimi.decode(audio_codes)  # [1, 1, samples]
                    audio_path = output_dir / f"response_sample_{b}.wav"
                    torchaudio.save(str(audio_path), pcm[0].cpu(), mimi.sample_rate)
            print(f"Audio saved to {output_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()

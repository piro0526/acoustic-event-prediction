# UR-FUNNY V2 Dataset

## Overview

UR-FUNNY is the first multimodal dataset for humor detection, created from TED Talk videos. The dataset captures humor through three modalities: **text** (words), **vision** (gestures), and **audio** (prosodic cues). V2 is an improved version of the original UR-FUNNY dataset, with noisy and overlapping data instances removed and more context sentences added.

## Source

- **Paper:** Hasan, M. K., Rahman, W., Zadeh, A., Zhong, J., Tanveer, I., Morency, L.-P., & Hoque, M. E. (2019). *UR-FUNNY: A Multimodal Language Dataset for Understanding Humor.* Proceedings of EMNLP-IJCNLP 2019, Hong Kong.
- **arXiv:** https://arxiv.org/abs/1904.06618
- **ACL Anthology:** https://aclanthology.org/D19-1211/
- **GitHub:** https://github.com/ROC-HCI/UR-FUNNY
- **Affiliation:** ROC HCI Lab (University of Rochester) & Language Technologies Institute (CMU)

## Dataset Statistics

| Item | Count |
|------|-------|
| TED Talk videos | 1,741 talks (1,866 videos) |
| Humorous punchlines | 8,257 |
| Non-humorous excerpts | 8,257 |
| Total utterances | 9,588 |
| Average context duration | 14.7 seconds |
| Average punchline duration | 4.58 seconds |

## Task

与えられた文の系列（コンテキスト + パンチライン）と、それに対応する視覚・音響モダリティから、パンチライン直後に笑いが発生するかどうかを二値分類で予測する。

負例はユーモアのある動画と同じ動画から抽出されており、ドメインの違いによるバイアスを防いでいる。

## Directory Structure

```
urfunny2/
├── README.md
├── metadata/
│   ├── data_folds.pkl
│   ├── humor_label_sdk.pkl
│   ├── language_sdk.pkl
│   ├── covarep_features_sdk.pkl
│   └── openface_features_sdk.pkl
└── videos/                       # 10,166 video clips (.mp4)
```

## Metadata Details

すべてのメタデータは Python pickle 形式 (`.pkl`) で保存されている。

### data_folds.pkl

Train / dev / test の分割情報を格納した辞書。各キー (`train`, `dev`, `test`) に対応する値は、ビデオセグメントIDのリスト。分割は話者独立 (speaker-independent) かつ均質 (homogeneous) に設計されている。

```python
{
    "train": [id_1, id_2, ...],
    "dev":   [id_1, id_2, ...],
    "test":  [id_1, id_2, ...]
}
```

### humor_label_sdk.pkl

各ビデオセグメントIDに対するユーモアの二値ラベルを格納した辞書。`1` = パンチライン後に笑いが発生（ユーモアあり）、`0` = 笑いなし（ユーモアなし）。

```python
{
    segment_id: 1 or 0,
    ...
}
```

### language_sdk.pkl

各ビデオセグメントIDに対するテキスト情報（トランスクリプト）を格納した辞書。単語インデックスが言語特徴量として使用される。

```python
{
    segment_id: {
        "punchline": [...],   # パンチラインの単語系列
        "context": [...]      # コンテキスト文の単語系列
    },
    ...
}
```

### covarep_features_sdk.pkl

COVAREP で抽出された音響特徴量を格納した辞書。各セグメントにはパンチラインとコンテキストの特徴量が含まれる。特徴量の次元数は **81**。

```python
{
    segment_id: {
        "punchline_features": np.ndarray,  # shape: [T, 81]
        "context_features": [np.ndarray, ...]
    },
    ...
}
```

### openface_features_sdk.pkl

OpenFace2 で抽出された視覚特徴量を格納した辞書。顔のAction Unit、頭部姿勢、視線方向などを含む。特徴量の次元数は **371**。

```python
{
    segment_id: {
        "punchline_features": np.ndarray,  # shape: [T, 371]
        "context_features": [np.ndarray, ...]
    },
    ...
}
```

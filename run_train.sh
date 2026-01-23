# !/bin/sh

nohup uv run torchrun --nproc_per_node=4 train_laughter_predictor.py \
        --transformer_dir output/transformer_outs \
        --labels_dir data/PodcastFillers/metadata/episode_laughter_prediction_intervals \
        --turns_dir data/PodcastFillers/metadata/episode_laughter_turns \
        --output_dir output/laughter_prediction_filtered \
        --batch_size 512 \
        --learning_rate 3e-4 \
        --epochs 100 \
        --num_workers 0 \
        --loss_type bce \
> filtered.out
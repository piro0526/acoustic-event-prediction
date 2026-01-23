# !/bin/sh

nohup uv run torchrun --nproc_per_node=8 scripts/train_laughter_predictor.py \
        --features_dir outputs/features_masked_concat \
        --shift_frames 1 \
        --output_dir outputs/laughter_prediction_masked \
        --batch_size 512 \
        --learning_rate 2e-4 \
        --epochs 50 \
        --early_stopping_patience 50 \
        --num_workers 4 \
        --loss_type focal \
> masked.out

uv run scripts/train_laughter_predictor.py \
        --features_dir outputs/features_masked_concat \
        --shift_frames 1 \
        --output_dir outputs/laughter_prediction_masked \
        --batch_size 512 \
        --learning_rate 2e-4 \
        --epochs 50 \
        --early_stopping_patience 50 \
        --num_workers 1 \
        --loss_type focal
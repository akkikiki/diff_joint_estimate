NUM_UNLABELED=10
PYTHONHASHSEED=0
python3 src/models/train_model.py \
    --lr 0.005 \
    --activation relu \
    --nhidden 32 \
    --dropout 0.0 \
    --epochs 500 \
    --seed 3 \
    --word-features length freq glove \
    --doc-features length bert_avg


# training
# TODO: Update with the tuned hyperparameters
rm data/interim/bigcn/logs_unlabeled_mod*.json

for MOD in 100000 10000 1000 100
do
  SAMPLED_EFCAMDAT="data/processed/efcamdat/EF201403_selection8.tokenized.mod${MOD}.jsonl"
  awk -v MOD="$MOD" 'NR%MOD==0' data/processed/efcamdat/EF201403_selection8.tokenized.jsonl \
  > $SAMPLED_EFCAMDAT

  python3 src/models/train_model.py \
      --lr 0.0001 \
      --nhidden 100 \
      --dropout 0.0 \
      --epochs 300 \
      --seed 3 \
      --featureless \
      --efcamdat_file_path $SAMPLED_EFCAMDAT > data/interim/bigcn/logs_unlabeled_mod${MOD}.json
done

# concatenate JSON outputs & visualize learning curve
cat data/interim/bigcn/logs_unlabeled_mod*.json | \
python3 src/visualization/unlabeled_learning_curve.py

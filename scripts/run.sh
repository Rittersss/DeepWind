export CUDA_VISIBLE_DEVICES=0

model_name=HeterogeneousModel

python -u run.py \
  --model $model_name \
  --LOSSFACTOR 0 \
  --QUANTILE 0.5 \
  --dataset track3_train.pkl \
  --DATA_PATH dataset \
  --MODEL_SAVE_PATH checkpoints/basemodel \
  --RESULT_SAVE_PATH test_results/basemodel \
  --num_epochs 10000 \
  --early_stop 100 \
  --learning_rate 0.001  \
  --feature_size 86 \
  --hidden_size 512 \
  --output_size 4 \
  --num_layers 1 \
  --timestep 48 \

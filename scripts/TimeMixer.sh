export CUDA_VISIBLE_DEVICES=0

model_name=MICN

seq_len=128
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=5
patience=10
batch_size=16

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/stock/AAPL\
  --data_path hours.csv \
  --model_id Stock_$seq_len'_'96 \
  --model $model_name \
  --data stock \
  --features S \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 32 \
  --e_layers $e_layers \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1\
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --target close \
  --batch_size 64 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \

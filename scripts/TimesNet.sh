export CUDA_VISIBLE_DEVICES=0

model_name=MICN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/stock/AAPL_minutes \
  --data_path hours.csv \
  --model_id Stock_iTransformer_96 \
  --model $model_name \
  --data stock \
  --features S \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 120 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --target close \
  --itr 1 \
  --train_epochs 5
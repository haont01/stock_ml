export CUDA_VISIBLE_DEVICES=0

model_name=Nonstationary_Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/stock/AAPL_minutes \
  --data_path train.csv \
  --model_id Stock_iTransformer_96 \
  --model $model_name \
  --data stock \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 120 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --target close \
  --itr 1 \
  --batch_size 64 \
  --train_epochs 5 \
  # --freq T
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/stock/top10v2 \
  --data_path train.csv \
  --model_id TimeMixer_96_'_'96 \
  --model TimeMixer \
  --data multi_stock \
  --features S \
  --seq_len 64 \
  --label_len 48 \
  --pred_len 1 \
  --e_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1\
  --des 'Exp' \
  --itr 1 \
  --d_model 16 \
  --d_ff 32 \
  --learning_rate 0.001 \
  --train_epochs 1 \
  --patience 10 \
  --target close \
  --batch_size 64 \
  --down_sampling_layers 2 \
  --down_sampling_method avg \
  --down_sampling_window 1 \
  --use_gpu 1\
  --factor 4 \
  --gpu 0 \
  --predict_multi_stock
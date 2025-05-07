docker run -it -v $(pwd):/app ps_llms \
  python /app/scripts/train_models.py \
  --model_name_or_path allenai/led-base-16384 \
  --train_file /app/data/postprocessed-data/mimic-iv-note-ext-di-bhc/dataset/train_4000_600_chars_20_examples.json \
  --val_file /app/data/postprocessed-data/mimic-iv-note-ext-di-bhc/dataset/valid_4000_600_chars_10_examples.json \
  --output_dir /app/scripts/model_ckpts \
  --batch_size 4 \
  --epochs 3 \
  --max_source_length 4096 \
  --fp16
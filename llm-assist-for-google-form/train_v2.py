import os
from datasets import load_dataset
import evaluate
import torch
from transformers import (
    LEDForConditionalGeneration,
    LEDTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# Paths
DATA_PATH = "/app/data/postprocessed-data/mimic-iv-note-ext-di-bhc/dataset/train_100.json"
OUTPUT_BASE = "./training_outputs"

# Load and split dataset
raw = load_dataset('json', data_files={'data': DATA_PATH})
splits = raw['data'].train_test_split(test_size=0.1, seed=42)
train_ds = splits['train']
eval_ds = splits['test']

# Load ROUGE metric
rouge = evaluate.load("rouge")

# Preprocessing function for LED
def preprocess_led(examples, tokenizer, max_input_length=16384, max_target_length=256):
    inputs = examples['text']
    targets = examples['summary']
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding='max_length'
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding='max_length'
        )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Compute ROUGE metrics
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    metrics = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    gen_lens = [len(pred.split()) for pred in decoded_preds]
    metrics['gen_len'] = sum(gen_lens) / len(gen_lens)
    return metrics

# Only use LED-base to fit within CPU memory
led_models = [
    ('allenai/led-base-16384', 'led-base')
]

for model_name, short_name in led_models:
    print(f"\n=== {short_name} Training & Evaluation (LoRA) ===")

    # Load tokenizer and base model
    tokenizer = LEDTokenizerFast.from_pretrained(model_name)
    model = LEDForConditionalGeneration.from_pretrained(model_name)

    # Enable checkpointing to save activation memory
    model.gradient_checkpointing_enable()

    # Configure and attach LoRA adapters
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params} / {total_params} total")

    # Tokenize datasets
    tokenized_train = train_ds.map(
        lambda ex: preprocess_led(ex, tokenizer),
        batched=True,
        remove_columns=['text', 'summary']
    )
    tokenized_eval = eval_ds.map(
        lambda ex: preprocess_led(ex, tokenizer),
        batched=True,
        remove_columns=['text', 'summary']
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Training arguments with Adafactor
    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(OUTPUT_BASE, short_name),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=3e-5,
        optim='adafactor',
        adafactor=True,
        num_train_epochs=3,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=4,
        dataloader_pin_memory=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Evaluate
    print("\n--- Evaluation Results ---")
    metrics = trainer.evaluate()
    for k, v in metrics.items(): print(f"{k}: {v}")

    # Save the adapter and tokenizer
    model.save_pretrained(os.path.join(OUTPUT_BASE, short_name))
    tokenizer.save_pretrained(os.path.join(OUTPUT_BASE, short_name))

print("\nLED-base trained and evaluated with LoRA.")

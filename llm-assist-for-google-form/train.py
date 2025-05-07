#!/usr/bin/env python
"""
train.py: Train and evaluate LED or Llama-2 models with optional W&B logging.

Simplified Usage:
  python train.py \
    --train_data train.json --val_data val.json [--test_data test.json] \
    --model led-base [--use_wandb]

Arguments:
  --train_data (required)  Path to training JSON/JSONL file
  --val_data   (required)  Path to validation JSON/JSONL file
  --test_data  (optional) Path to test JSON/JSONL file for final evaluation
  --model      (required)  One of [led-base, led-long, llama7b, llama70b]
  --use_wandb  (flag)      If set, logs to W&B under project name equal to model

The JSON files must contain records with keys 'text' and 'summary'.
"""
import argparse
import json
import torch
from datasets import Dataset
from transformers import (
    LEDTokenizerFast,
    LEDForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    LlamaTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
import evaluate

def load_json_file(path):
    records = []
    with open(path, 'r') as f:
        first = ''
        while True:
            c = f.read(1)
            if not c or not c.isspace(): first = c; break
        f.seek(0)
        if first == '[':
            records = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return Dataset.from_dict({
        'document': [r['text'] for r in records],
        'summary':  [r['summary'] for r in records]
    })

def split_train_val(path, test_size=0.2, seed=42):
    ds = load_json_file(path)
    return ds.train_test_split(test_size=test_size, seed=seed)

def make_compute_metrics(tokenizer):
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')
    def compute_metrics(eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(preds, tuple): preds = preds[0]
        dec_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dec_labels= tokenizer.batch_decode(labels, skip_special_tokens=True)
        r = rouge.compute(predictions=dec_preds, references=dec_labels)
        b = bertscore.compute(predictions=dec_preds, references=dec_labels, lang='en')
        return {
            'rouge1': r['rouge1'].mid.fmeasure,
            'rouge2': r['rouge2'].mid.fmeasure,
            'rougeL': r['rougeL'].mid.fmeasure,
            'bertscore_f1': sum(b['f1'])/len(b['f1'])
        }
    return compute_metrics

def preprocess_led(batch, tokenizer, win, tgt):
    inp = tokenizer(batch['document'], max_length=win, truncation=True, padding='max_length')
    lab = tokenizer(batch['summary'],  max_length=tgt, truncation=True, padding='max_length')
    inp['labels'] = lab['input_ids']
    return inp

def train_led(train_ds, val_ds, args):
    if args.use_wandb:
        import wandb
        wandb.init(project=args.model)
    tokenizer = LEDTokenizerFast.from_pretrained(args.ckpt)
    model     = LEDForConditionalGeneration.from_pretrained(args.ckpt, from_tf=True)
    tok_train = train_ds.map(
        lambda b: preprocess_led(b, tokenizer, args.max_input_len, args.max_target_len),
        batched=True, remove_columns=['document','summary']
    )
    tok_val   = val_ds.map(
        lambda b: preprocess_led(b, tokenizer, args.max_input_len, args.max_target_len),
        batched=True, remove_columns=['document','summary']
    )
    report_to = ['wandb'] if args.use_wandb else []
    tr_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.model}_output",
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=True,
        predict_with_generate=True,
        logging_steps=args.log_steps,
        report_to=report_to,
        run_name=args.model if args.use_wandb else None
    )
    trainer = Seq2SeqTrainer(
        model=model, args=tr_args,
        train_dataset=tok_train, eval_dataset=tok_val,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer)
    )
    trainer.train()
    return trainer, tokenizer, model

def train_llama(train_ds, val_ds, args):
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt, padding_side='left')
    model     = LlamaForCausalLM.from_pretrained(
        args.ckpt, load_in_8bit=True, device_map='auto'
    )
    lora_cfg = LoraConfig(
        r=8, lora_alpha=32, target_modules=['q_proj','v_proj'],
        lora_dropout=0.05, bias='none', task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora_cfg)
    def tok_fn(b):
        txt = 'Summarize: ' + b['document'] + '\nSummary:'
        return tokenizer(txt, truncation=True, max_length=args.max_input_len)
    tok_train = train_ds.map(tok_fn, batched=False)
    tok_val   = val_ds.map(tok_fn, batched=False)
    tr_args = TrainingArguments(
        output_dir=f"{args.model}_output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=True,
        optim='paged_adamw_8bit',
        logging_steps=args.log_steps,
        save_total_limit=2,
        **({'deepspeed':'ds_config.json'} if '70b' in args.ckpt else {})
    )
    trainer = Trainer(
        model=model, args=tr_args,
        train_dataset=tok_train, eval_dataset=tok_val,
        tokenizer=tokenizer,
        data_collator=lambda s: {'input_ids': torch.stack([x['input_ids'] for x in s])}
    )
    trainer.train()
    return trainer, tokenizer, model

def evaluate_preds(model, tokenizer, ds, args):
    rouge = evaluate.load('rouge')
    bert  = evaluate.load('bertscore')
    docs, sums = list(ds['document']), list(ds['summary'])
    preds, refs = [], []
    for i in range(0, len(docs), args.eval_bs):
        batch = docs[i:i+args.eval_bs]
        inp = tokenizer(batch, return_tensors='pt', padding=True,
                        truncation=True, max_length=args.max_input_len).to(model.device)
        out = model.generate(**inp, max_new_tokens=args.max_target_len)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds += decoded; refs += sums[i:i+args.eval_bs]
    r = rouge.compute(predictions=preds, references=refs)
    b = bert.compute(predictions=preds, references=refs, lang='en')
    print(f"ROUGE1 {r['rouge1'].mid.fmeasure:.4f}, ROUGE2 {r['rouge2'].mid.fmeasure:.4f}, "
          f"ROUGEL {r['rougeL'].mid.fmeasure:.4f}, BERT-F1 {sum(b['f1'])/len(b['f1']):.4f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_data', required=True, help='Path to training JSON/JSONL')
    p.add_argument('--val_data',   required=True, help='Path to validation JSON/JSONL')
    p.add_argument('--test_data',  help='Path to test JSON/JSONL')
    p.add_argument('--model',      required=True,
                   choices=['led-base','led-long','llama7b','llama70b'],
                   help='Model to train')
    p.add_argument('--epochs',     type=int,   default=3)
    p.add_argument('--bs',         type=int,   default=2)
    p.add_argument('--accum',      type=int,   default=8)
    p.add_argument('--lr',         type=float, default=3e-5)
    p.add_argument('--log_steps',  type=int,   default=100)
    p.add_argument('--max_input_len',  type=int, default=16384)
    p.add_argument('--max_target_len', type=int, default=512)
    p.add_argument('--eval_bs',     type=int,   default=8)
    p.add_argument('--use_wandb',   action='store_true', help='Log to W&B under project named after model')
    args = p.parse_args()
    # dataset loading
    train_val = split_train_val(args.train_data) if not args.val_data else None
    train_ds  = train_val['train'] if not args.val_data else load_json_file(args.train_data)
    val_ds    = train_val['test']  if not args.val_data else load_json_file(args.val_data)
    test_ds   = load_json_file(args.test_data) if args.test_data else None
    # checkpoint map
    ckpts = {
        'led-base':'allenai/led-base-16384',
        'led-long':'allenai/led-large-16384',
        'llama7b':'meta-llama/Llama-2-7b-hf',
        'llama70b':'meta-llama/Llama-2-70b-hf'
    }
    args.ckpt = ckpts[args.model]
    # training
    if args.model in ['led-base','led-long']:
        trainer, tokenizer, model = train_led(train_ds, val_ds, args)
    else:
        trainer, tokenizer, model = train_llama(train_ds, val_ds, args)
    # evaluation
    print("\n=== Validation Metrics ===")
    evaluate_preds(model, tokenizer, val_ds, args)
    if test_ds:
        print("\n=== Test Metrics ===")
        evaluate_preds(model, tokenizer, test_ds, args)

if __name__ == '__main__':
    main()

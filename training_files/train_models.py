from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import argparse
import torch

class CustomTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=torch.cuda.is_available()
        )

def main(args):
    # Load tokenizer and model
    import wandb
    wandb.init(settings=wandb.Settings(init_timeout=180))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    # Load dataset
    data_files = {"train": args.train_file, "validation": args.val_file}
    raw_datasets = load_dataset("json", data_files=data_files)

    # Tokenize
    def preprocess(batch):
        # Tokenize source texts
        model_inputs = tokenizer(
            batch["text"],
            max_length=args.max_source_length,
            truncation=True,
            padding="max_length",
        )

        # Tokenize target texts
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["summary"],
                max_length=args.max_target_length,
                truncation=True,
                padding="max_length",
            )

        # Replace pad_token_id with -100 to ignore in loss
        labels_input_ids = []
        for label_seq in labels["input_ids"]:
            label_seq = [(token if token != tokenizer.pad_token_id else -100) for token in label_seq]
            labels_input_ids.append(label_seq)

        model_inputs["labels"] = labels_input_ids
        return model_inputs

    tokenized_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=1,
        eval_steps=None,
        save_strategy="no",
        fp16=args.fp16,
        report_to="wandb",
        run_name=args.output_dir,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    print(f"Number of training examples: {len(tokenized_datasets['train'])}")
    print(f"Train batch size: {training_args.per_device_train_batch_size}")
    print(f"Num steps: {len(tokenized_datasets['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs}")
    
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_source_length", type=int, default=4096)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    main(args)
    print("Script completed")
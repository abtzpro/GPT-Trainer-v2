import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import os


class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized = self.tokenizer.encode_plus(self.texts[idx], max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': torch.tensor(tokenized['input_ids']), 'attention_mask': torch.tensor(tokenized['attention_mask'])}


def prepare_data(file_path, test_size=0.1, max_length=100):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().splitlines()

    train_texts, val_texts = train_test_split(data, test_size=test_size, random_state=42)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_dataset = CustomDataset(train_texts, tokenizer, max_length)
    val_dataset = CustomDataset(val_texts, tokenizer, max_length)

    config = GPT2Config.from_pretrained("gpt2")
    training_args = TrainingArguments(output_dir="output",
                                      overwrite_output_dir=True,
                                      per_device_train_batch_size=4,
                                      per_device_eval_batch_size=4,
                                      num_train_epochs=3,
                                      learning_rate=5e-5,
                                      fp16=True,
                                      logging_steps=100,
                                      save_steps=500,
                                      evaluation_strategy="steps")

    return train_dataset, val_dataset, config, training_args


def train_model(train_dataset, val_dataset, config, training_args):
    model = GPT2LMHeadModel(config)
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      tokenizer=GPT2Tokenizer.from_pretrained("gpt2"))
    trainer.train()
    return model


def evaluate_model(model, tokenizer):
    input_text = "What is the meaning of life?"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    train_dataset, val_dataset, config, training_args = prepare_data("data.txt")
    save_path = "config.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"config": config.to_dict(), "training_args": training_args.to_dict()}, f, indent=4)

    model = train_model(train_dataset, val_dataset, config, training_args)
    tokenizer = GPT2Tokenizer.from_pretrained("output")
    response = evaluate_model(model, tokenizer)
    print("Generated response:", response)

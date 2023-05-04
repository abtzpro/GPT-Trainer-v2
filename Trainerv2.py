import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import os

# Dataset Preparation
def collect_and_preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().splitlines()
    return data

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

def split_dataset(texts, test_size=0.1):
    train_texts, val_texts = train_test_split(texts, test_size=test_size, random_state=42)
    return train_texts, val_texts

# Model Configuration
def choose_pretrained_model(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return tokenizer

def configure_model(model_name="gpt2", n_gpu=1, lr=5e-5, batch_size=4, num_train_epochs=3):
    config = GPT2Config.from_pretrained(model_name)
    training_args = TrainingArguments(output_dir="output",
                                      overwrite_output_dir=True,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      num_train_epochs=num_train_epochs,
                                      learning_rate=lr,
                                      fp16=n_gpu > 0,
                                      logging_steps=100,
                                      save_steps=500,
                                      evaluation_strategy="steps")
    return config, training_args

def save_configuration(config, training_args, save_path="config.json"):
    with open(save_path, "w") as f:
        json.dump({"config": config.to_dict(), "training_args": training_args.to_dict()}, f, indent=4)

# Training
def initialize_model(config):
    model = GPT2LMHeadModel(config)
    return model

def setup_training_loop(tokenizer, train_dataset, val_dataset, model, training_args):
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      tokenizer=tokenizer)
    return trainer

def train_model(trainer):
    trainer.train()

# Evaluation
def evaluate_model(trainer):
    eval_results = trainer.evaluate()
    return eval_results

# Usage
def load_finetuned_model(model_name="output"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def generate_response(tokenizer, model, input_text, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def integrate_chatgpt_model(tokenizer, model, input_text):
    response = generate_response(tokenizer, model, input_text)
    return response
if __name__ == "__main__":
    file_path = "data.txt"
    texts = collect_and_preprocess_data(file_path)
    train_texts, val_texts = split_dataset(texts)

    model_name = "gpt2"
    tokenizer = choose_pretrained_model(model_name)
    max_length = 100

    train_dataset = CustomDataset(train_texts, tokenizer, max_length)
    val_dataset = CustomDataset(val_texts, tokenizer, max_length)

    config, training_args = configure_model(model_name=model_name)
    save_configuration(config, training_args)

    model = initialize_model(config)
    trainer = setup_training_loop(tokenizer, train_dataset, val_dataset, model, training_args)

    train_model(trainer)
    eval_results = evaluate_model(trainer)
    print("Evaluation results:", eval_results)

    # Load the fine-tuned model
    tokenizer, model = load_finetuned_model()

    # Example usage of the integrated ChatGPT model
    input_text = "What is the meaning of life?"
    response = integrate_chatgpt_model(tokenizer, model, input_text)
    print("Generated response:", response)

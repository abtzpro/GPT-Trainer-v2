# GPT-Trainer-v2

This script is designed to train a GPT-2 language model on a custom dataset and generate responses to input text. It utilizes the Hugging Face Transformers library and PyTorch.

## Functions

The script provides the following functions:

- `prepare_data(file_path, test_size=0.1, max_length=100)`: reads in a text file from `file_path`, splits the data into training and validation sets, tokenizes the data using the GPT-2 tokenizer, and returns the resulting datasets, model configuration, and training arguments.
- `train_model(train_dataset, val_dataset, config, training_args)`: initializes a GPT-2 language model using the provided `config`, trains the model on the provided datasets using the provided `training_args`, and returns the trained model.
- `evaluate_model(model, tokenizer)`: generates a response to the input text "What is the meaning of life?" using the provided `model` and `tokenizer`.
- `save_configuration(config, training_args, save_path="config.json")`: saves the provided model configuration and training arguments to a JSON file located at `save_path`.
- `load_configuration(load_path="config.json")`: loads model configuration and training arguments from a JSON file located at `load_path`.
- `load_finetuned_model(model_name="output")`: loads a finetuned GPT-2 language model and its tokenizer from the specified directory `model_name`.
- `generate_response(tokenizer, model, input_text, max_length=100)`: generates a response to the provided `input_text` using the provided `model` and `tokenizer`, with a maximum length of `max_length`.
- `integrate_chatgpt_model(tokenizer, model, input_text)`: integrates the `generate_response` function and returns the resulting response to the provided `input_text`.

## Usage

1. Ensure that `data.txt` is located in the same directory as the script.
2. To train the GPT-2 language model, run the script with no arguments: `python GPT-Trainer-v2.py`. This will split the data into training and validation sets, train the model using default parameters, and save the resulting configuration to `config.json` and the finetuned model to a directory called `output`.
3. To load the finetuned model and generate a response to input text, use the following code:

```python
from GPT-Trainer-v2 import load_finetuned_model, generate_response

tokenizer, model = load_finetuned_model()
input_text = "What is the meaning of life?"
response = generate_response(tokenizer, model, input_text)
print("Generated response:", response)
```

## Developed By

This script was developed by Adam Rivers and Hello Security LLC with the support of OpenAI (I am still waiting on a job offer :( sniffle).

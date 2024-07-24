import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=1024)

args = parser.parse_args()

def apply_chat_template(batch):
    return {"formatted_chat": [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in batch["conversation"]]}

def tokenized_dataset(dataset):
    return dataset.map(tokenize_function,batched=True,remove_columns=['formatted_chat'])

def tokenize_function(dataset):
    return tokenizer(dataset['formatted_chat'],padding=True,max_length=args.max_length,truncation=True)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM-135M-Instruct',torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M-Instruct')
    dataset = load_dataset('benchang1110/emoji-chat', split='train')
    dataset = dataset.map(apply_chat_template, batched=True)
    dataset = tokenized_dataset(dataset)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    
    
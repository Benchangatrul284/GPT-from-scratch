import datasets
from datasets import load_dataset

def convert_conversational(sample):
    conversation = []
    conversation.append({"content": sample['prompt'],"role": "user"})
    conversation.append({"content": sample['chosen'],"role": "assistant"})
    return {'conversation':conversation}

dataset = load_dataset("Orion-zhen/dpo-mathinstuct-emoji",split='train')
dataset = dataset.map(convert_conversational,remove_columns=['prompt','chosen','rejected'])
dataset.push_to_hub('benchang1110/emoji-chat',token='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')
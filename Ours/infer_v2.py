import argparse
import os
import re

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def detect_entity_type(text):
    """Detect entity type using heuristics."""
    text = str(text).strip()

    # Check if date (YYYY-MM-DD, YYYY, etc.) - check before numeric
    if re.match(r'^\d{4}(-\d{2}(-\d{2})?)?$', text):
        return 'DATE'

    # Check if year range (e.g., "1990-2000")
    if re.match(r'^\d{4}-\d{4}$', text):
        return 'DATE'

    # Check if numeric
    try:
        float(text.replace(',', ''))
        return 'NUM'
    except:
        pass

    # Default to entity
    return 'ENTITY'


def build_input(subject, obj):
    """Enhanced input with entity type markers."""
    subj_type = detect_entity_type(subject)
    obj_type = detect_entity_type(obj)
    return f"[E1:{subj_type}] {subject} [/E1] [E2:{obj_type}] {obj} [/E2]"


class InferDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = build_input(str(row['Subject']), str(row['Object']))
        enc = self.tokenizer(text, max_length=self.max_length, padding='max_length',
                             truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }


class CPAModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.labels_path, encoding='utf-8') as f:
        classes = [l.strip() for l in f if l.strip()]
    id2label = {i: c for i, c in enumerate(classes)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = CPAModel(args.model_name, len(classes), dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    df = pd.read_csv(args.input_csv, low_memory=False, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]

    loader = DataLoader(InferDataset(df, tokenizer, args.max_length),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="inferring"):
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out_df = df[['Subject', 'Object']].copy()
    out_df['Label'] = [id2label[p] for p in preds]
    out_df.to_csv(args.output_file, index=False, encoding='utf-8')
    print(f"saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', default='../dataset/test.csv')
    parser.add_argument('--labels_path', default='../dataset/labels.txt')
    parser.add_argument('--model_name', default='microsoft/deberta-v3-base')
    parser.add_argument('--model_dir', default='./cpa_output/best')
    parser.add_argument('--model_path', default='./cpa_output/best/best_model.pt')
    parser.add_argument('--output_file', default='./result/submission_v2.csv')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    run_inference(args)

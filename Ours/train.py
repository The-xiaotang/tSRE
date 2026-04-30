import argparse
import os
import random
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def load_data(dir_path):
    all_data = []
    for filename in tqdm(os.listdir(dir_path), desc="loading"):
        if not filename.endswith('.csv'):
            continue
        label = filename[:-4]
        try:
            df = pd.read_csv(os.path.join(dir_path, filename), low_memory=False, encoding='utf-8-sig')
            df.columns = [c.strip() for c in df.columns]
            if 'Subject' in df.columns and 'Object' in df.columns:
                df = df[['Subject', 'Object']].dropna()
                df['label'] = label
                all_data.append(df)
        except Exception as e:
            logging.warning(f"{filename}: {e}")
    df = pd.concat(all_data, ignore_index=True)
    df['Subject'] = df['Subject'].astype(str)
    df['Object'] = df['Object'].astype(str)
    return df


def build_input(subject, obj):
    """Input enhancement: structured template with entity type hints."""
    return f"relation: [subject] {subject} [object] {obj}"


class CPADataset(Dataset):
    def __init__(self, df, tokenizer, label_encoder, max_length):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le = label_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = build_input(row['Subject'], row['Object'])
        enc = self.tokenizer(text, max_length=self.max_length, padding='max_length',
                             truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.le.transform([row['label']])[0], dtype=torch.long),
        }


class CPAModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


def compute_class_weights(df, label_encoder):
    """Compute m_weights per competition scoring formula."""
    counts = df['label'].value_counts()
    counts_max = counts.max()
    counts_min = counts.min()
    weights = {}
    for label, cnt in counts.items():
        w = (counts_max - cnt + counts_min * 0.1) / (counts_max + counts_min * 0.1)
        weights[label] = max(w, 1e-6)
    # Map to label ids
    weight_tensor = torch.zeros(len(label_encoder.classes_))
    for label, w in weights.items():
        idx = label_encoder.transform([label])[0]
        weight_tensor[idx] = w
    return weight_tensor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'cpa_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO,
        handlers=[logging.FileHandler(os.path.join(save_dir, 'train.log'), encoding='utf-8'),
                  logging.StreamHandler()])
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"device: {device}")

    df = load_data(args.train_dir)
    le = LabelEncoder()
    le.fit(df['label'].unique())
    with open(os.path.join(save_dir, 'label_classes.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(le.classes_))

    counts = df['label'].value_counts()
    rare = counts[counts < 2].index
    df_rare = df[df['label'].isin(rare)]
    df_common = df[~df['label'].isin(rare)]
    train_c, val_df = train_test_split(df_common, test_size=args.val_ratio,
                                       stratify=df_common['label'], random_state=args.seed)
    train_df = pd.concat([train_c, df_rare]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    logging.info(f"train={len(train_df)}, val={len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = DataLoader(CPADataset(train_df, tokenizer, le, args.max_length),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(CPADataset(val_df, tokenizer, le, args.max_length),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CPAModel(args.model_name, len(le.classes_)).to(device)

    # Weighted loss using competition scoring weights
    class_weights = compute_class_weights(train_df, le).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(total_steps * args.warmup_ratio),
                                                num_training_steps=total_steps)

    best_acc, patience_counter = 0.0, 0
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            tr_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                preds = logits.argmax(dim=1)
                correct += (preds == batch['label'].to(device)).sum().item()
                total += len(batch['label'])
        val_acc = correct / total
        logging.info(f"Epoch {epoch+1} | loss={tr_loss/len(train_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            tokenizer.save_pretrained(save_dir)
            logging.info(f"saved best model (acc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info("early stop")
                break

    logging.info(f"done. best_acc={best_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='../dataset/Train_Set')
    parser.add_argument('--output_dir', default='./cpa_output')
    parser.add_argument('--model_name', default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    args = parser.parse_args()
    run_training(args)

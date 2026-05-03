import argparse
import os
import random
import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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


class FocalLoss(nn.Module):
    """Focal Loss for handling hard samples."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def compute_class_weights(df, label_encoder):
    """Compute weights per competition scoring formula."""
    counts = df['label'].value_counts()
    counts_max = counts.max()
    counts_min = counts.min()
    weights = {}
    for label, cnt in counts.items():
        w = (counts_max - cnt + counts_min * 0.1) / (counts_max + counts_min * 0.1)
        weights[label] = max(w, 1e-6)
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
    if args.resume and args.resume_dir:
        save_dir = args.resume_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(args.output_dir, f'cpa_v2_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO,
        handlers=[logging.FileHandler(os.path.join(save_dir, 'train.log'), encoding='utf-8', mode='a'),
                  logging.StreamHandler()])
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"device: {device}, model: {args.model_name}")

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

    # Oversample small classes
    if args.oversample_threshold > 0:
        train_counts = train_df['label'].value_counts()
        small_classes = train_counts[train_counts < args.oversample_threshold].index
        oversampled = []
        for label in small_classes:
            df_small = train_df[train_df['label'] == label]
            count = len(df_small)
            repeat = max(1, args.oversample_threshold // count)
            oversampled.extend([df_small] * repeat)
        if oversampled:
            train_df = pd.concat([train_df] + oversampled).sample(frac=1, random_state=args.seed).reset_index(drop=True)
            logging.info(f"oversampled {len(small_classes)} classes to >={args.oversample_threshold} samples")

    logging.info(f"train={len(train_df)}, val={len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = DataLoader(CPADataset(train_df, tokenizer, le, args.max_length),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(CPADataset(val_df, tokenizer, le, args.max_length),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CPAModel(args.model_name, len(le.classes_), dropout=args.dropout).to(device)

    class_weights = compute_class_weights(train_df, le).to(device)
    if args.use_focal_loss:
        loss_fn = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        logging.info(f"Using Focal Loss (gamma={args.focal_gamma})")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        logging.info(f"Using CrossEntropyLoss (label_smoothing={args.label_smoothing})")

    if args.use_rdrop:
        logging.info(f"Using R-Drop (alpha={args.rdrop_alpha})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(total_steps * args.warmup_ratio),
                                                num_training_steps=total_steps)

    # Resume from checkpoint if exists
    start_epoch, best_acc, patience_counter = 0, 0.0, 0
    ckpt_path = os.path.join(save_dir, 'checkpoint.pt')
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        patience_counter = ckpt['patience_counter']
        logging.info(f"resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        tr_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            if args.use_rdrop:
                # R-Drop: forward twice with different dropout
                logits1 = model(input_ids, mask)
                logits2 = model(input_ids, mask)

                # CE loss
                ce_loss = (loss_fn(logits1, labels) + loss_fn(logits2, labels)) / 2

                # KL divergence between two outputs
                p1 = F.log_softmax(logits1, dim=-1)
                p2 = F.log_softmax(logits2, dim=-1)
                kl_loss = F.kl_div(p1, F.softmax(logits2, dim=-1), reduction='batchmean')
                kl_loss += F.kl_div(p2, F.softmax(logits1, dim=-1), reduction='batchmean')
                kl_loss /= 2

                loss = ce_loss + args.rdrop_alpha * kl_loss
            else:
                logits = model(input_ids, mask)
                loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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

        # Save checkpoint for resume
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'patience_counter': patience_counter,
        }, ckpt_path)

    logging.info(f"done. best_acc={best_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='../dataset/Train_Set')
    parser.add_argument('--output_dir', default='./cpa_output')
    parser.add_argument('--model_name', default='microsoft/deberta-v3-base')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--oversample_threshold', type=int, default=20)
    parser.add_argument('--use_rdrop', action='store_true')
    parser.add_argument('--rdrop_alpha', type=float, default=0.5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_dir', type=str, default=None)
    args = parser.parse_args()
    run_training(args)

import argparse
import os
import random
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from paddlenlp.transformers import AutoTokenizer, AutoModel, LinearDecayWithWarmup

# data load
def load_data_from_directory(dir_path):
    all_data = []
    if not os.path.exists(dir_path):
        raise ValueError(f"can't find: {dir_path}")

    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    logging.info(f"load data from {dir_path} ...")

    for filename in tqdm(csv_files, desc=f"loading {os.path.basename(dir_path)}"):
        file_path = os.path.join(dir_path, filename)
        label_name = filename[:-4]
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')
            if df.empty:
                continue
            df.columns = [str(col).strip() for col in df.columns]
            if 'Subject' in df.columns and 'Object' in df.columns:
                df = df[['Subject', 'Object']].dropna()
                df['label'] = label_name
                all_data.append(df)
        except Exception as e:
            logging.warning(f"{filename} load error: {e}")

    if not all_data:
        raise ValueError(f"{dir_path} not valid data")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Subject'] = full_df['Subject'].astype(str)
    full_df['Object'] = full_df['Object'].astype(str)
    return full_df

# encode data
def encode_text(tokenizer, text_input, max_length):
    try:
        encoding = tokenizer(
            text_input,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
    except TypeError:
        encoding = tokenizer(
            text=text_input,
            max_seq_len=max_length,
            pad_to_max_seq_len=True,
            truncation=True,
            return_attention_mask=True,
        )

    input_ids = encoding['input_ids']
    attention_mask = encoding.get('attention_mask', None)

    if attention_mask is None:
        seq_len = encoding.get('seq_len', len(input_ids))
        seq_len = min(seq_len, max_length)
        attention_mask = [1] * seq_len + [0] * (max_length - seq_len)

    return np.array(input_ids, dtype='int64'), np.array(attention_mask, dtype='int64')

# dataset
class RelationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_encoder, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le = label_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_input = f"{row['Subject']} [SEP] {row['Object']}"
        input_ids, attention_mask = encode_text(self.tokenizer, text_input, self.max_length)
        label_id = self.le.transform([row['label']])[0]
        return {
            'valid': True,
            'token_ids': input_ids,
            'cls_mask': attention_mask,
            'label_id': np.int64(label_id),
        }

def dynamic_collate_fn(samples):
    valid_samples = [s for s in samples if s.get('valid', False)]
    if not valid_samples:
        return None

    return {
        'data': np.stack([s['token_ids'] for s in valid_samples]).astype('int64'),
        'label': np.array([s['label_id'] for s in valid_samples], dtype='int64'),
        'cls_mask': np.stack([s['cls_mask'] for s in valid_samples]).astype('int64'),
    }

# model
class CPAModel(nn.Layer):
    def __init__(self, model_name, num_labels, use_flash_attn=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        hidden_size = None
        if hasattr(self.encoder, 'config'):
            if isinstance(self.encoder.config, dict):
                hidden_size = self.encoder.config.get('hidden_size', None)
            else:
                hidden_size = getattr(self.encoder.config, 'hidden_size', None)

        if hidden_size is None and hasattr(self.encoder, 'embeddings') and hasattr(self.encoder.embeddings, 'word_embeddings'):
            hidden_size = self.encoder.embeddings.word_embeddings.weight.shape[-1]

        if hidden_size is None:
            raise ValueError('hidden_size in None')

        self.classifier = nn.Linear(hidden_size, num_labels)

        if use_flash_attn:
            logging.warning('flash attention not activate')

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs

        cls_embedding = sequence_output[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits

# seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

# log
def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'train.log'), mode='w', encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )

# device
def resolve_device(device_arg):
    try:
        custom_types = paddle.device.get_all_custom_device_type()
    except Exception:
        custom_types = []

    logging.info(f'custom device types: {custom_types}')
    
    if device_arg:
        try:
            dev = paddle.set_device(device_arg)
            logging.info(f'use: {dev}')
            return dev
        except Exception as e:
            logging.warning(f'{device_arg} use error: {e}')
            
    dev = paddle.set_device('cpu')
    logging.warning('set device to CPU')
    return dev

# labels
def save_label_classes(label_encoder, save_dir):
    path = os.path.join(save_dir, 'label_classes.txt')
    with open(path, 'w', encoding='utf-8') as f:
        for label in label_encoder.classes_:
            f.write(f'{label}\n')

# train
def run_training(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'cpa_{timestamp}')
    setup_logging(save_dir)
    set_seed(args.random_seed)
    device = resolve_device(args.device)

    logging.info(f'device: {device}')

    # 1. load train data
    raw_train_df = load_data_from_directory(args.train_dir)

    # 2. bulid lable
    label_encoder = LabelEncoder()
    label_encoder.fit(raw_train_df['label'].unique())
    num_classes = len(label_encoder.classes_)
    logging.info(f'label_num: {num_classes}')
    save_label_classes(label_encoder, save_dir)

    # 3. split dataset
    counts = raw_train_df['label'].value_counts()
    rare_labels = counts[counts < 2].index
    df_rare = raw_train_df[raw_train_df['label'].isin(rare_labels)]
    df_common = raw_train_df[~raw_train_df['label'].isin(rare_labels)]

    if len(df_common) == 0:
        raise ValueError("data num < 2, can't split dataset")

    train_c, val_c = train_test_split(
        df_common,
        test_size=args.val_ratio,
        stratify=df_common['label'],
        random_state=args.random_seed,
    )
    train_df = pd.concat([train_c, df_rare]).sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    val_df = val_c.reset_index(drop=True)
    logging.info(f'split success: train={len(train_df)}, val={len(val_df)}')

    # 4. Tokenizer & DataLoader
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    train_loader = DataLoader(
        RelationDataset(train_df, tokenizer, label_encoder, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dynamic_collate_fn,
        num_workers=args.num_workers,
        return_list=True,
    )
    val_loader = DataLoader(
        RelationDataset(val_df, tokenizer, label_encoder, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dynamic_collate_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    # 5. model init
    model = CPAModel(args.shortcut_name, num_classes, args.use_flash_attention)
    total_steps = max(1, len(train_loader) * args.epoch)
    lr_scheduler = LinearDecayWithWarmup(args.lr, total_steps, warmup=args.warmup_ratio)
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    use_amp = args.use_amp and str(device) != 'cpu'
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024) if use_amp else None

    # 6. training
    best_acc = 0.0
    patience_counter = 0
    patience_limit = args.patience

    logging.info('start training...')
    for epoch in range(args.epoch):
        model.train()
        tr_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epoch}')
        for batch in pbar:
            if batch is None:
                continue

            input_ids = paddle.to_tensor(batch['data'], dtype='int64')
            mask = paddle.to_tensor(batch['cls_mask'], dtype='int64')
            label_ids = paddle.to_tensor(batch['label'], dtype='int64')

            if use_amp:
                with paddle.amp.auto_cast(enable=True):
                    logits = model(input_ids, mask)
                    loss = loss_fn(logits, label_ids)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()
            else:
                logits = model(input_ids, mask)
                loss = loss_fn(logits, label_ids)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

            lr_scheduler.step()
            loss_value = float(loss.numpy())
            tr_loss += loss_value
            train_steps += 1
            pbar.set_postfix({'loss': f'{loss_value:.4f}'})

        # val stage
        model.eval()
        val_correct, val_total = 0, 0
        with paddle.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                input_ids = paddle.to_tensor(batch['data'], dtype='int64')
                mask = paddle.to_tensor(batch['cls_mask'], dtype='int64')
                label_ids = paddle.to_tensor(batch['label'], dtype='int64')

                if use_amp:
                    with paddle.amp.auto_cast(enable=True):
                        logits = model(input_ids, mask)
                else:
                    logits = model(input_ids, mask)

                preds = paddle.argmax(logits, axis=1)
                val_correct += int((preds == label_ids).astype('int64').sum().numpy())
                val_total += int(label_ids.shape[0])

        avg_train_loss = tr_loss / max(1, train_steps)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        logging.info(f'Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            paddle.save(model.state_dict(), os.path.join(save_dir, 'best_model.pdparams'))
            try:
                tokenizer.save_pretrained(save_dir)
            except Exception:
                pass
            logging.info(f'best model! (Acc: {best_acc:.4f})')
        else:
            patience_counter += 1
            logging.info(f'early stop count: {patience_counter}/{patience_limit}')
            if patience_counter == patience_limit:
                logging.info(f'{patience_limit} epoch not up, early stop!!!')
                break

    logging.info(f'train finish, best acc: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default="./dataset/Train_Set")
    parser.add_argument('--output_dir', type=str, default='./cpa_output')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_flash_attention', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='gpu')
    args = parser.parse_args()
    run_training(args)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# ================= CONFIG =================
MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 128
BATCH_SIZE = 4
DEVICE = "cpu"
TARGETS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar',
'conventions']

# ================= DATASET =================
class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# ================= MODEL =================
class EssayScorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(self.backbone.config.hidden_size,
len(TARGETS))

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids,
attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # CLS token
        return self.head(self.dropout(pooled))

# ================= LOAD DATA =================
df = pd.read_csv("train (1).csv")
for c in TARGETS:
    df[c] = df[c].astype(float)

from sklearn.model_selection import train_test_split
_, val_df = train_test_split(df, test_size=0.1, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
val_dataset = EssayDataset(val_df.full_text.values,
val_df[TARGETS].values, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================= LOAD MODEL =================
model = EssayScorer().to(DEVICE)
model.load_state_dict(torch.load("best_essay_scorer_cpu.pt",
map_location=DEVICE))
model.eval()

# ================= OPTIONAL VALIDATION (kept but no printing)
preds, trues = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids, attention_mask)
        preds.append(outputs.cpu().numpy())
        trues.append(labels.cpu().numpy())

preds = np.concatenate(preds)
trues = np.concatenate(trues)

# ==========================================================
# ===============     USER INPUT PREDICTION     ============
# ==========================================================

print("\nModel loaded. You can now enter essays to get predicted scores.")
print("Type 'exit' to quit.\n")

def predict_user_essay(text):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prediction = output.cpu().numpy().flatten()

    return prediction

while True:
    essay = input("\nPaste your essay (or type 'exit'): ").strip()
    if essay.lower() == "exit":
        print("Exiting...")
        break

    scores = predict_user_essay(essay)
    print("\nPredicted Scores:")
    for t, s in zip(TARGETS, scores):
        print(f"  {t}: {s:.2f}")
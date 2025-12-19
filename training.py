import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# ================= CONFIG =================
MODEL_NAME = "microsoft/deberta-v3-small"  # fast and CPU-friendly
MAX_LENGTH = 128  # shorter sequence → faster
BATCH_SIZE = 4  # small batch for CPU
EPOCHS = 1  # fewer epochs → faster
DEVICE = "cpu"
TARGETS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar',
           'conventions']

# ================= LOAD DATA =================
df = pd.read_csv("train.csv")
for c in TARGETS:
    df[c] = df[c].astype(float)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

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

# ================= SETUP =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EssayScorer().to(DEVICE)

train_dataset = EssayDataset(train_df.full_text.values,
                             train_df[TARGETS].values, tokenizer)
val_dataset = EssayDataset(val_df.full_text.values,
                           val_df[TARGETS].values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

optimizer = AdamW(model.parameters(), lr=3e-5)
criterion = torch.nn.MSELoss()

# ================= TRAIN =================
print("Training started...")
best_mcrmse = 99.0

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    train_losses = []
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        loop.set_postfix(train_loss=np.mean(train_losses))

    # ================= VALIDATION =================
    model.eval()
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

    # Mean column-wise RMSE
    mcrmse = np.mean([np.sqrt(mean_squared_error(trues[:, i], preds[:, i]))
                      for i in range(len(TARGETS))])

    print(f"Epoch {epoch+1} - Train Loss: {np.mean(train_losses):.4f} - "
          f"Val MCRMSE: {mcrmse:.4f}")

    if mcrmse < best_mcrmse:
        best_mcrmse = mcrmse
        torch.save(model.state_dict(), "best_essay_scorer_cpu.pt")
        print("New best model saved!")

print(f"\nDONE! Final MCRMSE: {best_mcrmse:.4f}")
print("Model saved as: best_essay_scorer_cpu.pt")

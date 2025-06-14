import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.autoencoder import SSTAutoencoder

# ========== Configuration ==========
PATCH_PATH = Path("data/patches/sst_patches_64x64.npy")
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Dataset Wrapper ==========
class SSTPatchDataset(Dataset):
    def __init__(self, patch_file):
        patches = np.load(patch_file)  # shape: (N, 64, 64)
        patches = np.nan_to_num(patches, nan=0.0)
        self.data = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)  # (N, 1, 64, 64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # x, target = same (autoencoder)

# ========== Load Data ==========
print("📦 Loading data ...")
dataset = SSTPatchDataset(PATCH_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== Model ==========
print("🧠 Initializing model ...")
model = SSTAutoencoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ========== Training Loop ==========
print("🏋️ Training started ...")
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"📈 Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(loader):.6f}")

# ========== Save Model ==========
torch.save(model.state_dict(), "outputs/sst_autoencoder.pth")
print("✅ Training complete. Model saved.")


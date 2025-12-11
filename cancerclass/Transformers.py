"""
FIXED: Cancer Classification with Transformer
Key fixes:
1. Feature selection applied (12,553 → 500 proteins)
2. Class weights for balance
3. Lower learning rate
4. Gradient clipping
5. Better hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime, timedelta
import pickle

# ============================================
# Setup
# ============================================

os.makedirs('progress', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================
# Model Definition
# ============================================

class CancerClassificationTransformer(nn.Module):
    """
    Transformer for cancer classification from protein expression
    """
    
    def __init__(self, n_proteins, n_classes, d_model=128, n_heads=8, 
                 n_layers=4, dropout=0.1):
        super().__init__()
        
        self.n_proteins = n_proteins
        self.n_classes = n_classes
        self.d_model = d_model
        
        # Project each protein to d_model dimensions
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(n_proteins, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        """
        Args:
            x: Protein expression tensor of shape (batch_size, n_proteins)
        Returns:
            logits: Class predictions of shape (batch_size, n_classes)
        """
        batch_size, n_proteins = x.shape
        
        # Reshape and project
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :n_proteins, :].to(x.device)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

# ============================================
# Dataset
# ============================================

class ProteinDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ============================================
# Load Data
# ============================================

print("="*70)
print("LOADING DATA")
print("="*70)

proteomics = pd.read_csv('77_cancer_proteomes_CPTAC_itraq.csv')
clinical = pd.read_csv('clinical_data_breast_cancer.csv')

sample_cols = proteomics.columns[3:-3].tolist()
X_raw = proteomics[sample_cols].T

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print(f"Raw features: {X_scaled.shape}")

# Extract labels
def extract_tcga_id(sample_name):
    if 'TCGA' in sample_name:
        patient_id = sample_name.replace('.TCGA', '').split('.')[0]
        return f"TCGA-{patient_id}"
    return None

sample_names = X_raw.index.tolist()
y_labels = []
matched_indices = []

for i, sample in enumerate(sample_names):
    tcga_id = extract_tcga_id(sample)
    if tcga_id in clinical['Complete TCGA ID'].values:
        idx = clinical[clinical['Complete TCGA ID'] == tcga_id].index[0]
        label = clinical.loc[idx, 'PAM50 mRNA']
        y_labels.append(label)
        matched_indices.append(i)

X_matched = X_scaled[matched_indices]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)

print(f"Matched samples: {len(y_labels)}")
print(f"Classes: {label_encoder.classes_}")

# Check class distribution
class_counts = Counter(y_encoded)
for i, name in enumerate(label_encoder.classes_):
    print(f"  {name}: {class_counts[i]} samples")

# ============================================
# FIX 1: APPLY FEATURE SELECTION!
# ============================================

print("\n" + "="*70)
print("FEATURE SELECTION")
print("="*70)

# Split data FIRST
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_matched, y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)

# Then apply feature selection
n_features = 500  # Reduce from 12,553 to 500!
selector = SelectKBest(f_classif, k=n_features)
X_train = selector.fit_transform(X_train_full, y_train)
X_test = selector.transform(X_test_full)

print(f"Features: {X_train_full.shape[1]} → {n_features}")
print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# ============================================
# FIX 2: CLASS WEIGHTS
# ============================================

n_samples = len(y_train)
n_classes = len(label_encoder.classes_)
class_weights = []

for class_idx in range(n_classes):
    count = (y_train == class_idx).sum()
    weight = n_samples / (n_classes * count)
    class_weights.append(weight)

weight_tensor = torch.FloatTensor(class_weights).to(device)
print(f"\nClass weights: {[f'{w:.2f}' for w in class_weights]}")

# ============================================
# Create Datasets
# ============================================

train_dataset = ProteinDataset(X_train, y_train)  # Now only 500 features!
test_dataset = ProteinDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ============================================
# Initialize Model
# ============================================

print("\n" + "="*70)
print("INITIALIZING MODEL")
print("="*70)

model = CancerClassificationTransformer(
    n_proteins=n_features,  # ← FIX: Use selected features (500), not all (12,553)!
    n_classes=n_classes,
    d_model=128,
    n_heads=8,
    n_layers=4,
    dropout=0.3  # ← Increased dropout
).to(device)

criterion = nn.CrossEntropyLoss(weight=weight_tensor)  # ← FIX: Use class weights
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # ← FIX: Lower learning rate

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# Training Loop with Fixes
# ============================================

print("\n" + "="*70)
print("TRAINING")
print("="*70)

n_epochs = 100
best_acc = 0
patience = 20
patience_counter = 0

train_losses = []
train_accs = []
test_accs = []
epochs_list = []

training_start = time.time()
epoch_times = []

for epoch in range(n_epochs):
    epoch_start = time.time()
    
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # FIX 3: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
    
    train_acc = train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    test_correct = 0
    test_total = 0
    test_predictions = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
    
    test_acc = test_correct / test_total
    
    # Track metrics
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    epochs_list.append(epoch + 1)
    
    # Calculate ETA
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = n_epochs - (epoch + 1)
    eta_seconds = avg_epoch_time * remaining_epochs
    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
    
    # Print progress with debugging info
    if (epoch + 1) % 10 == 0 or epoch < 5:
        unique_preds = len(set(test_predictions))
        print(f'Epoch [{epoch+1:3d}/{n_epochs}] '
              f'Loss: {avg_train_loss:.4f} '
              f'Train: {train_acc:.3f} '
              f'Test: {test_acc:.3f} '
              f'(uses {unique_preds}/{n_classes} classes) '
              f'| ETA: {eta_time.strftime("%H:%M:%S")}')
    
    scheduler.step(test_acc)
    
    # Early stopping
    if test_acc > best_acc:
        best_acc = test_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_cancer_model.pt')
        if (epoch + 1) % 10 == 0:
            print(f'  ✓ New best: {best_acc:.3f}')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

print(f'\nBest Test Accuracy: {best_acc:.1%}')

# ============================================
# Save Everything
# ============================================

final_checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'label_encoder_classes': label_encoder.classes_,
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
    'selector': selector,  # ← IMPORTANT: Save feature selector!
    'best_acc': best_acc,
    'model_config': {
        'n_proteins': n_features,  # ← 500, not 12,553
        'n_classes': n_classes,
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'dropout': 0.3,
    },
}

torch.save(final_checkpoint, 'final_model_complete.pt')

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)

print("✓ All models and preprocessors saved")

# ============================================
# Final Evaluation
# ============================================

print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

model.load_state_dict(torch.load('best_cancer_model.pt'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

predicted_names = label_encoder.inverse_transform(all_preds)
true_names = label_encoder.inverse_transform(all_labels)

print("\nClassification Report:")
print(classification_report(true_names, predicted_names))

# ============================================
# Visualizations
# ============================================

print("\nGenerating plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_list, train_accs, 'b-', label='Train Accuracy', linewidth=2)
ax2.plot(epochs_list, test_accs, 'r-', label='Test Accuracy', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Plots saved to: training_curves.png")

print("\n" + "="*70)
print(f"TRAINING COMPLETE - Best Accuracy: {best_acc:.1%}")
print("="*70)

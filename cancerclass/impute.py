import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout_rate=0.3):
        super(VariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=dropout_rate) 
        
        self.fc_mu = nn.Linear(512, latent_dim)      
        self.fc_logvar = nn.Linear(512, latent_dim)  
        
        self.fc3 = nn.Linear(latent_dim, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop3 = nn.Dropout(p=dropout_rate)      
        self.fc4 = nn.Linear(512, input_dim)

    def encode(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
        h1 = self.drop1(h1) 
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.bn3(self.fc3(z)))
        h3 = self.drop3(h3) 
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
def vae_loss_function(recon_x, x, mu, logvar, mask):
    # MSE
    MSE = F.mse_loss(recon_x[mask], x[mask], reduction='sum')
    #  KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + (0.01 * KLD)

def run_vae_imputer(df_input, epochs=100,dropout_rate=0.3):
    print("   Running Variational Autoencoder (VAE)...")
    scaler = MinMaxScaler()
    data_filled = df_input.fillna(0).values
    data_scaled = scaler.fit_transform(data_filled)
    
    n_samples, n_features = data_scaled.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize VAE
    model = VariationalAutoencoder(input_dim=n_features, latent_dim=64,dropout_rate=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002) 
    
    inputs = torch.tensor(data_scaled, dtype=torch.float32).to(device)
    # Create mask: True where data exists
    mask_tensor = torch.tensor(~df_input.isna().values, dtype=torch.bool).to(device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inputs)
        # Calculate custom VAE loss
        loss = vae_loss_function(recon_batch, inputs, mu, logvar, mask_tensor)
        loss.backward()
        optimizer.step()
        
    # Generate Final Imputations
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(inputs)
        recon_batch = model.decode(mu).cpu().numpy()
        
    data_vae = scaler.inverse_transform(recon_batch)
    return pd.DataFrame(data_vae, columns=df_input.columns, index=df_input.index)

class DreamAIImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=10, trees_estimators=20, vae_epochs=100):
        self.n_neighbors = n_neighbors
        self.trees_estimators = trees_estimators
        self.vae_epochs = vae_epochs
        
        # Initialize sub-imputers
        self.knn = KNNImputer(n_neighbors=n_neighbors)
        self.trees = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=trees_estimators, n_jobs=1, random_state=42),
            max_iter=5, n_nearest_features=50, random_state=42, verbose=0
        )

    def fit(self, X, y=None):

        self.knn.fit(X)
        self.trees.fit(X)
        return self

    def transform(self, X):
        print(f"   [DreamAI] Running Ensemble on shape {X.shape}...")
        
        # Convert to DataFrame 
        if isinstance(X, np.ndarray):
            df_X = pd.DataFrame(X)
        else:
            df_X = X.copy()

        # 1: KNN
        X_knn = self.knn.transform(X)
        
        # 2: ExtraTrees
        X_trees = self.trees.transform(X)
        
        # 3: VAE
        df_vae_filled = run_vae_imputer(df_X, epochs=self.vae_epochs, dropout_rate=0.3)
        X_vae = df_vae_filled.values 

        # 5. Aggregate (Average)
        X_final = (X_knn + X_trees + X_vae) / 3
        
        return X_final
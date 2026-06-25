import json
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. THE CORRECTED CVAE-GMM MODEL DEFINITION
# ==========================================
class EV_DayAhead_CVAE(nn.Module):
    def __init__(self, target_dim=3, context_dim=5, latent_dim=3, dictionary_dim=40, num_clusters_K=10, xi=1e-5):
        super().__init__()
        self.T = target_dim        
        self.C = context_dim       
        self.L = latent_dim        
        self.V = dictionary_dim   
        self.K = num_clusters_K
        self.xi = xi               
        
        # Pattern Dictionary Matrix (U) per cluster
        self.U = nn.Parameter(torch.randn(self.K, self.T, self.V))
        
        # --- Encoder Network: q(z | x, c) ---
        self.encoder = nn.Sequential(
            nn.Linear(self.T + self.C, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2)
        )
        self.enc_mu = nn.Linear(512, self.L)
        self.enc_logvar = nn.Linear(512, self.L)
        
        # --- Decoder Network: f(z, c) -> GMM Parameters ---
        self.decoder = nn.Sequential(
            nn.Linear(self.L + self.C, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2)
        )
        # Output parameters for ALL K clusters simultaneously
        self.dec_mu = nn.Linear(512, self.K * self.T)
        self.dec_tilde_sigma = nn.Linear(512, self.K * self.V) 
        self.dec_pi_logits = nn.Linear(512, self.K)

    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=-1)
        h = self.encoder(inputs)
        return self.enc_mu(h), self.enc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def map_latent_to_gmm_components(self, z, c):
        batch_size = z.size(0)
        decoder_input = torch.cat([z, c], dim=-1)
        h = self.decoder(decoder_input)
        
        # Reshape outputs to [batch_size, K, ...]
        mu_k = self.dec_mu(h).view(batch_size, self.K, self.T)
        
        tilde_sigma = F.softplus(self.dec_tilde_sigma(h)).view(batch_size, self.K, self.V)
        tilde_sigma = torch.clamp(tilde_sigma, min=1e-3, max=1e1)
        
        # Construct covariance matrices for each cluster component
        Sigma_k = []
        I_noise = torch.eye(self.T, device=z.device).unsqueeze(0).unsqueeze(0) * self.xi
        
        for k in range(self.K):
            diag_sigma2 = torch.diag_embed(tilde_sigma[:, k, :] ** 2)
            U_k = self.U[k].unsqueeze(0).expand(batch_size, -1, -1)
            cov_pattern = torch.bmm(U_k, torch.bmm(diag_sigma2, U_k.transpose(1, 2)))
            cov_pattern = 0.5 * (cov_pattern + cov_pattern.transpose(1, 2))
            Sigma_k.append(cov_pattern)
            
        Sigma_k = torch.stack(Sigma_k, dim=1) + I_noise # [batch_size, K, T, T]
        
        pi_logits = self.dec_pi_logits(h)
        pi_k = F.softmax(pi_logits, dim=-1) # [batch_size, K]
        
        return mu_k, Sigma_k, pi_k

    def forward(self, x, c):
        mu_z, logvar_z = self.encode(x, c)
        z = self.reparameterize(mu_z, logvar_z)
        mu_k, Sigma_k, pi_k = self.map_latent_to_gmm_components(z, c)
        return mu_k, Sigma_k, pi_k, mu_z, logvar_z

    def generate_day_ahead_scenarios(self, single_day_c, num_scenarios_S=25):
        self.eval()
        device = next(self.parameters()).device
        c = single_day_c.to(device) 
        
        with torch.no_grad():
            # Sample a common latent anchor representing the day's structural demand
            z = torch.randn(1, self.L, device=device) 
            mu_k, Sigma_k, pi_k = self.map_latent_to_gmm_components(z, c)
            
            # Extract GMM profiles
            GMM_means = mu_k.squeeze(0)          
            GMM_covs = Sigma_k.squeeze(0)  
            GMM_probabilities = pi_k.squeeze(0)
            
            # Sample which cluster index each scenario belongs to based on the learned GMM weights
            chosen_cluster_indices = torch.multinomial(GMM_probabilities, num_scenarios_S, replacement=True)
            
            scenarios = []
            for s in range(num_scenarios_S):
                k_selected = chosen_cluster_indices[s].item()
                mu_s = GMM_means[k_selected]
                Sigma_s = GMM_covs[k_selected]
                
                # Robust MVN Sampling
                try:
                    mvn = torch.distributions.MultivariateNormal(loc=mu_s, covariance_matrix=Sigma_s)
                    simulated_session = mvn.sample()
                except RuntimeError:
                    # Fallback to mean if covariance matrix becomes unstable during generation
                    simulated_session = mu_s 
                            
                scenarios.append((k_selected, simulated_session))
                
        return scenarios, GMM_means, GMM_covs, GMM_probabilities


# ==========================================
# 2. FIXED GMM-ELBO LOSS FUNCTION
# ==========================================
def cvae_elbo_loss(x, mu_k, Sigma_k, pi_k, mu_z, logvar_z, beta=1.0):
    batch_size = x.size(0)
    K = pi_k.size(1)
    
    # Compute log probabilities for each sample across ALL K components
    log_probs = []
    for k in range(K):
        try:
            mvn = torch.distributions.MultivariateNormal(loc=mu_k[:, k], covariance_matrix=Sigma_k[:, k])
            log_prob_k = mvn.log_prob(x) # [batch_size]
        except ValueError:
            # Numerical fallback to MSE if a matrix degenerates mid-batch
            log_prob_k = -F.mse_loss(mu_k[:, k], x, reduction='none').mean(dim=-1) * 10.0
        log_probs.append(log_prob_k)
        
    log_probs = torch.stack(log_probs, dim=1) # [batch_size, K]
    
    # Correct GMM Log-Likelihood: LogSumExp(log(pi_k) + log_prob_k)
    log_pi = torch.log(pi_k + 1e-8)
    recon_loss = -torch.logsumexp(log_pi + log_probs, dim=1).mean()
                
    # KL-Divergence Penalty
    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=-1).mean()
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ==========================================
# 3. DIRECTORY-SAFE DATA PREPROCESSING LAYER
# ==========================================
def preprocess_acndata(file_name):
    # 1. Look up paths safely across execution environments
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    path_relative_to_script = os.path.join(script_dir, file_name)
    path_relative_to_cwd = os.path.join(os.getcwd(), file_name)
    
    if os.path.exists(path_relative_to_script):
        file_path = path_relative_to_script
    elif os.path.exists(path_relative_to_cwd):
        file_path = path_relative_to_cwd
    else:
        print(f"[WARN] File '{file_name}' not found at:\n  -> {path_relative_to_script}\n  -> {path_relative_to_cwd}\nGenerating dummy simulation dataset...")
        X = np.random.randn(200, 3).astype(np.float32)
        C = np.random.randint(0, 2, size=(200, 5)).astype(np.float32)
        return torch.tensor(X), torch.tensor(C)
        
    print(f"[INFO] Successfully found dataset. Parsing profiles from: {file_path}")
    
    with open(file_path, "r") as f:
        data = json.load(f)
        
    sessions = data.get("_items", [])
    targets, contexts = [], []
    
    # Using a clean format string since your file contains standard GMT tags
    date_format = "%a, %d %b %Y %H:%M:%S GMT"
    
    for s in sessions:
        try:
            conn_str = s.get("connectionTime")
            disc_str = s.get("disconnectTime")
            kwh = s.get("kWhDelivered")
            
            if not conn_str or not disc_str or kwh is None: 
                continue
                
            # Parse times to datetime objects
            dt_arr = datetime.strptime(conn_str, date_format)
            dt_dep = datetime.strptime(disc_str, date_format)
            
            # Compute fractional hours
            arr_hour = dt_arr.hour + dt_arr.minute / 60.0
            
            # Account for stays spanning past midnight into the next day(s)
            days_spent = (dt_dep.date() - dt_arr.date()).days
            dep_hour = dt_dep.hour + dt_dep.minute / 60.0 + (days_spent * 24.0)
            
            # Data sanitation thresholds
            if kwh <= 0.1 or kwh > 100.0 or dep_hour <= arr_hour or (dep_hour - arr_hour) > 72.0: 
                continue 
            
            # Targets: [Arrival, Energy, Departure]
            targets.append([arr_hour, kwh, dep_hour])
            
            # Cyclical Context features (Sine/Cosine embeddings for time of year and week)
            is_weekday = 1.0 if dt_arr.weekday() < 5 else 0.0
            contexts.append([
                is_weekday,
                np.sin(2 * np.pi * dt_arr.month / 12),
                np.cos(2 * np.pi * dt_arr.month / 12),
                np.sin(2 * np.pi * dt_arr.weekday() / 7),
                np.cos(2 * np.pi * dt_arr.weekday() / 7)
            ])
        except Exception as e:
            # Silently skip anomalies or single corrupted logs
            continue

    if len(targets) == 0:
        raise ValueError(f"CRITICAL: Found file at {file_path}, but 0 sessions passed parsing. Check date formats or key values.")

    print(f"[SUCCESS] Loaded {len(targets)} valid historical charging sessions.")
    return torch.tensor(np.array(targets, dtype=np.float32)), torch.tensor(np.array(contexts, dtype=np.float32))


# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Force Python to look inside the exact absolute folder path
    FILE_NAME = r"C:\Users\Matteo\Desktop\Ancillary Services\Code\EV_data_analysis\American_Data\acndata_sessions.json"
    
    X_raw, C_data = preprocess_acndata(FILE_NAME)
        
    X_mean, X_std = X_raw.mean(dim=0, keepdim=True), X_raw.std(dim=0, keepdim=True)
    X_normalized = (X_raw - X_mean) / (X_std + 1e-8)

    dataset = TensorDataset(X_normalized, C_data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = EV_DayAhead_CVAE(target_dim=3, context_dim=5, latent_dim=3, dictionary_dim=10, num_clusters_K=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n--- PHASE A: RUNNING OFFLINE NETWORK WEIGHT OPTIMIZATION ---")
    for epoch in range(1, 31):
        model.train()
        epoch_loss = 0.0
        beta = min(1.0, epoch / 15.0) # Linear annealing
        
        for batch_x, batch_c in train_loader:
            batch_x, batch_c = batch_x.to(device), batch_c.to(device)
            optimizer.zero_grad()
            
            mu_k, Sigma_k, pi_k, mu_z, logvar_z = model(batch_x, batch_c)
            loss, _, _ = cvae_elbo_loss(batch_x, mu_k, Sigma_k, pi_k, mu_z, logvar_z, beta=beta)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/30 | Beta: {beta:.3f} | Total Loss: {epoch_loss / len(X_raw):7.4f}")
        
    print("\n--- PHASE B: EXECUTING DAY-AHEAD INFERENCE DISTRIBUTION ---")
    tomorrow_context = torch.tensor([[1.0, np.sin(2*np.pi*6/12), np.cos(2*np.pi*6/12), 0.0, 1.0]], dtype=torch.float32)
    
    scenarios, cluster_means_norm, _, cluster_probabilities = model.generate_day_ahead_scenarios(tomorrow_context, num_scenarios_S=10)
    cluster_means_real = (cluster_means_norm.cpu() * X_std) + X_mean
    
    print("\n" + "="*80)
    print("                      PRODUCTION EXPORT RESULTS OUTLINE                         ")
    print("" + "="*80)
    print(f"[METRIC] Learned Cluster Distribution Assignments:")
    for k in range(model.K):
        c_mu = cluster_means_real[k].numpy()
        print(f"  -> Cluster #{k:02d} | Probability Weight: {cluster_probabilities[k].item()*100:6.2f}% | Archetype Center: [Arr: {c_mu[0]%24:5.2f}h, Energy: {c_mu[1]:5.2f}kWh, Dep: {c_mu[2]%24:5.2f}h]")
        
    print(f"\n[OUTPUT] Explicit Profile Generation Tracking Samples:")
    for idx, (origin_cluster, normalized_sample) in enumerate(scenarios):
        real_sample = (normalized_sample.cpu() * X_std.squeeze(0)) + X_mean.squeeze(0)
        print(f"  -> Scenario #{idx+1:02d}: Sourced via [Cluster #{origin_cluster:02d}] | Arrival: {max(0., min(24., float(real_sample[0]))):5.2f}h | Energy: {max(0.1, float(real_sample[1])):5.2f}kWh | Departure: {float(real_sample[2])%24:5.2f}h")
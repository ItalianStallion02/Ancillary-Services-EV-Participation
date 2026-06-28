import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

#
# This file uses the cVAE-GMM to:
# - sample scenarios of fleet size for a given day
# - build clusters for each scenario
# - use stratified sampling to sample representative profiles from a given cluster and scale energy demand to match fleet size
#


#
# What comes before: cvae_cluster_sampling.py
#
# What comes after: NOTHING
#

# ==========================================
# 1. THE CVAE-GMM MODEL WITH STRATIFIED FLEET-GENERATION
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
        
        mu_k = self.dec_mu(h).view(batch_size, self.K, self.T)
        tilde_sigma = F.softplus(self.dec_tilde_sigma(h)).view(batch_size, self.K, self.V)
        tilde_sigma = torch.clamp(tilde_sigma, min=1e-3, max=1e1)
        
        Sigma_k = []
        I_noise = torch.eye(self.T, device=z.device).unsqueeze(0).unsqueeze(0) * self.xi
        
        for k in range(self.K):
            diag_sigma2 = torch.diag_embed(tilde_sigma[:, k, :] ** 2)
            U_k = self.U[k].unsqueeze(0).expand(batch_size, -1, -1)
            cov_pattern = torch.bmm(U_k, torch.bmm(diag_sigma2, U_k.transpose(1, 2)))
            cov_pattern = 0.5 * (cov_pattern + cov_pattern.transpose(1, 2))
            Sigma_k.append(cov_pattern)
            
        Sigma_k = torch.stack(Sigma_k, dim=1) + I_noise 
        pi_logits = self.dec_pi_logits(h)
        pi_k = F.softmax(pi_logits, dim=-1) 
        
        return mu_k, Sigma_k, pi_k

    def forward(self, x, c):
        mu_z, logvar_z = self.encode(x, c)
        z = self.reparameterize(mu_z, logvar_z)
        mu_k, Sigma_k, pi_k = self.map_latent_to_gmm_components(z, c)
        return mu_k, Sigma_k, pi_k, mu_z, logvar_z

    def generate_day_ahead_fleet_scenarios(self, single_day_c, num_vehicles_N, num_scenarios_S=5, agents_per_cluster=20):
        self.eval()
        device = next(self.parameters()).device
        c = single_day_c.to(device) 
        
        fleet_scenarios = []
        
        with torch.no_grad():
            for s in range(num_scenarios_S):
                z = torch.randn(1, self.L, device=device) 
                mu_k, Sigma_k, pi_k = self.map_latent_to_gmm_components(z, c)
                
                GMM_means = mu_k.squeeze(0)          
                GMM_covs = Sigma_k.squeeze(0)  
                GMM_probabilities = pi_k.squeeze(0)
                
                sampled_fleet_sessions = []
                vehicle_origin_clusters = []
                agent_weights = []
                
                # Apply Stratified Sampling across the cluster distributions
                for k in range(self.K):
                    pi_k_val = GMM_probabilities[k].item()
                    fleet_allocated_k = pi_k_val * num_vehicles_N
                    
                    mu_s = GMM_means[k]
                    Sigma_s = GMM_covs[k]
                    
                    try:
                        mvn = torch.distributions.MultivariateNormal(loc=mu_s, covariance_matrix=Sigma_s)
                    except RuntimeError:
                        mvn = None
                    
                    # Generate distinct representative agents for this archetype pocket
                    for a in range(agents_per_cluster):
                        if mvn is not None:
                            try:
                                simulated_session = mvn.sample()
                            except RuntimeError:
                                simulated_session = mu_s
                        else:
                            simulated_session = mu_s
                                
                        sampled_fleet_sessions.append(simulated_session)
                        vehicle_origin_clusters.append(k)
                        
                        # Scale factor: Each agent represents an equal share of this component's fleet density
                        agent_weights.append(fleet_allocated_k / agents_per_cluster)
                
                fleet_scenarios.append({
                    "scenario_idx": s + 1,
                    "cluster_weights": GMM_probabilities,
                    "cluster_means": GMM_means,
                    "fleet_tensor_normalized": torch.stack(sampled_fleet_sessions),
                    "fleet_origins": vehicle_origin_clusters,
                    "agent_weights": torch.tensor(agent_weights, dtype=torch.float32)
                })
                
        return fleet_scenarios


# ==========================================
# 2. GMM-ELBO LOSS FUNCTION
# ==========================================
def cvae_elbo_loss(x, mu_k, Sigma_k, pi_k, mu_z, logvar_z, beta=1.0):
    batch_size = x.size(0)
    K = pi_k.size(1)
    
    log_probs = []
    for k in range(K):
        try:
            mvn = torch.distributions.MultivariateNormal(loc=mu_k[:, k], covariance_matrix=Sigma_k[:, k])
            log_prob_k = mvn.log_prob(x) 
        except ValueError:
            log_prob_k = -F.mse_loss(mu_k[:, k], x, reduction='none').mean(dim=-1) * 10.0
        log_probs.append(log_prob_k)
        
    log_probs = torch.stack(log_probs, dim=1) 
    log_pi = torch.log(pi_k + 1e-8)
    recon_loss = -torch.logsumexp(log_pi + log_probs, dim=1).mean()
                
    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=-1).mean()
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ==========================================
# 3. DATA PREPROCESSING LAYER
# ==========================================
def preprocess_acndata_with_counts(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    path_relative_to_script = os.path.join(script_dir, file_name)
    path_relative_to_cwd = os.path.join(os.getcwd(), file_name)
    
    if os.path.exists(path_relative_to_script):
        file_path = path_relative_to_script
    elif os.path.exists(path_relative_to_cwd):
        file_path = path_relative_to_cwd
    else:
        print(f"[WARN] File '{file_name}' not found. Emulating dummy simulation data distributions...")
        X = np.random.randn(200, 3).astype(np.float32)
        C = np.random.randint(0, 2, size=(200, 5)).astype(np.float32)
        return torch.tensor(X), torch.tensor(C), [45, 52, 60], [4, 7, 5]
        
    print(f"[INFO] Processing profiles and daily registration historical distributions from: {file_path}")
    
    with open(file_path, "r") as f:
        data = json.load(f)
        
    sessions = data.get("_items", [])
    targets, contexts = [], []
    weekday_date_tracker = []
    weekend_date_tracker = []
    
    date_format = "%a, %d %b %Y %H:%M:%S GMT"
    gmt_tz = ZoneInfo("GMT")
    local_tz = ZoneInfo("America/Los_Angeles")
    
    for s in sessions:
        try:
            conn_str = s.get("connectionTime")
            disc_str = s.get("disconnectTime")
            kwh = s.get("kWhDelivered")
            
            if not conn_str or not disc_str or kwh is None: 
                continue
                
            dt_arr_gmt = datetime.strptime(conn_str, date_format).replace(tzinfo=gmt_tz)
            dt_dep_gmt = datetime.strptime(disc_str, date_format).replace(tzinfo=gmt_tz)
            dt_arr = dt_arr_gmt.astimezone(local_tz)
            dt_dep = dt_dep_gmt.astimezone(local_tz)
            
            local_date_str = dt_arr.strftime("%Y-%m-%d")
            arr_hour = dt_arr.hour + dt_arr.minute / 60.0
            days_spent = (dt_dep.date() - dt_arr.date()).days
            dep_hour = dt_dep.hour + dt_dep.minute / 60.0 + (days_spent * 24.0)
            
            if kwh <= 0.1 or kwh > 100.0 or dep_hour <= arr_hour or (dep_hour - arr_hour) > 72.0: 
                continue 
            
            targets.append([arr_hour, kwh, dep_hour])
            is_weekday = 1.0 if dt_arr.weekday() < 5 else 0.0
            
            if is_weekday == 1.0:
                weekday_date_tracker.append(local_date_str)
            else:
                weekend_date_tracker.append(local_date_str)
                
            contexts.append([
                is_weekday,
                np.sin(2 * np.pi * dt_arr.month / 12),
                np.cos(2 * np.pi * dt_arr.month / 12),
                np.sin(2 * np.pi * dt_arr.weekday() / 7),
                np.cos(2 * np.pi * dt_arr.weekday() / 7)
            ])
        except Exception:
            continue

    if len(targets) == 0:
        raise ValueError("CRITICAL: Failed to parse historical items.")

    weekday_counts = list(Counter(weekday_date_tracker).values())
    weekend_counts = list(Counter(weekend_date_tracker).values())
    
    print(f"[SUCCESS] Loaded {len(targets)} sessions across {len(weekday_counts)+len(weekend_counts)} unique tracking dates.")
    return torch.tensor(np.array(targets, dtype=np.float32)), torch.tensor(np.array(contexts, dtype=np.float32)), weekday_counts, weekend_counts


# ==========================================
# 4. EXECUTION PIPELINE WITH SAMPLING VERIFICATION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FILE_NAME = r"C:\Users\Matteo\Desktop\Ancillary Services\Code\EV_data_analysis\American_Data\acndata_sessions_jpl.json"
    
    X_raw, C_data, weekday_counts, weekend_counts = preprocess_acndata_with_counts(FILE_NAME)
        
    X_mean, X_std = X_raw.mean(dim=0, keepdim=True), X_raw.std(dim=0, keepdim=True)
    X_normalized = (X_raw - X_mean) / (X_std + 1e-8)

    dataset = TensorDataset(X_normalized, C_data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Instantiate with exactly 10 clusters
    model = EV_DayAhead_CVAE(target_dim=3, context_dim=5, latent_dim=3, dictionary_dim=10, num_clusters_K=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n--- PHASE A: MODEL OPTIMIZATION TRAINING ---")
    for epoch in range(1, 31):
        model.train()
        epoch_loss = 0.0
        beta = min(1.0, epoch / 15.0) 
        
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
        
    print("\n--- PHASE B: EXECUTING STRATIFIED SCALED DAY-AHEAD GENERATION ---")
    tomorrow_context = torch.tensor([[1.0, np.sin(2*np.pi*6/12), np.cos(2*np.pi*6/12), 0.0, 1.0]], dtype=torch.float32)
    is_weekday = tomorrow_context[0, 0].item()
    
    relevant_historical_counts = weekday_counts if is_weekday == 1.0 else weekend_counts
    poisson_lambda = np.mean(relevant_historical_counts)
    
    NUM_SCENARIOS = 3
    AGENTS_PER_CLUSTER = 20
    
    print("\n" + "="*95)
    print("               PRODUCTION STRATIFIED SCALED SCENARIO OUTPUTS W/ DETAILS                  ")
    print("="*95)
    
    for s in range(NUM_SCENARIOS):
        base_day_N = np.random.poisson(poisson_lambda)
        if base_day_N < 1: base_day_N = 1  
        
        FLEET_SCALE_FACTOR = 10000
        total_fleet_N = base_day_N * FLEET_SCALE_FACTOR
        
        day_scenario = model.generate_day_ahead_fleet_scenarios(
            tomorrow_context, 
            num_vehicles_N=total_fleet_N, 
            num_scenarios_S=1, 
            agents_per_cluster=AGENTS_PER_CLUSTER
        )[0]
        
        fleet_real = (day_scenario['fleet_tensor_normalized'].cpu() * X_std.squeeze(0)) + X_mean.squeeze(0)
        fleet_origins = np.array(day_scenario['fleet_origins'])
        agent_weights = day_scenario['agent_weights'].numpy()
        
        total_day_energy_request = (fleet_real[:, 1].numpy() * agent_weights).sum()
        
        print(f"\n>>>> [JOINT FLEET SCENARIO #{s+1:02d} | Represented Fleet Volume: {total_fleet_N:,} EVs] <<<<")
        print(f"  [METRIC] Projected Fleet Aggregate Energy Requirement: {total_day_energy_request:,.2f} kWh")
        print("  Allocated Volumetric Composition Per Driver Cluster Archetype:")
        
        cluster_means_real = (day_scenario['cluster_means'].cpu() * X_std) + X_mean
        
        # Determine an audit cluster dynamically for this scenario to show example variations
        # Let's print out the stratified sampling breakdown for Cluster #2 in Scenario 1, Cluster #3 in Scenario 2, etc.
        audit_cluster = (s + 2) % model.K
        
        for k in range(model.K):
            c_mu = cluster_means_real[k].numpy()
            pct_weight = day_scenario['cluster_weights'][k].item() * 100
            
            cluster_agent_indices = [i for i, origin in enumerate(fleet_origins) if origin == k]
            cluster_assigned_evs = agent_weights[cluster_agent_indices].sum()
            
            print(f"    -> Cluster #{k:02d}: Probability {pct_weight:5.1f}% | SCALED VOLUME: {cluster_assigned_evs:11,.1f} EVs | Center: [Arr: {c_mu[0]%24:5.2f}h, Energy: {c_mu[1]:5.2f}kWh, Dep: {c_mu[2]%24:5.2f}h]")
            
            # NEW MODIFICATION: Print the unique stratified variations inside the designated audit cluster
            if k == audit_cluster:
                print(f"       " + "-"*75)
                print(f"       [AUDIT] Printing Stratified Variations for Sampled Archetype Cluster #{k:02d}:")
                # Show up to 5 representative agents out of the 20 sampled inside this specific stratum
                for idx_count, agent_idx in enumerate(cluster_agent_indices[:10]):
                    agent_profile = fleet_real[agent_idx].numpy()
                    a_weight = agent_weights[agent_idx]
                    print(f"         * Rep Agent #{idx_count+1:02d} Profile -> Arrival: {agent_profile[0]%24:5.2f}h | Energy: {agent_profile[1]:5.2f} kWh | Departure: {agent_profile[2]%24:5.2f}h | (Represents: {a_weight:,.1f} EVs)")
                if len(cluster_agent_indices) > 5:
                    print(f"         ... ({len(cluster_agent_indices) - 5} remaining agent strata calculated for this pocket block) ...")
                print(f"       " + "-"*75)
import torch
import torch.nn as nn
from tqdm import tqdm
import snntorch as snn
from ..layers.neurons import ALIF, SwiGLU
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import random

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_topology_proof(loader, device, class_names, max_batches=100):
    
    print("Accumulating spatial averages per class...")
    
    num_classes = len(class_names)
    # Accumulators: [Class, Height, Width]
    # Assuming input is 32x32. Adjust if different.
    class_sums = torch.zeros(num_classes, 32, 32).to(device)
    class_counts = torch.zeros(num_classes).to(device)
    
    # Iterate through data
    for batch_idx, (data, _, target) in enumerate(loader):
        if batch_idx >= max_batches: break

        data = data.to(device)
        target = target.to(device)
        
        if data.dim() == 4: # [B, C, H, W]
            spatial_data = data.mean(dim=1) 
        elif data.dim() == 5: # [B, T, C, H, W]
            spatial_data = data.mean(dim=(1, 2))
        else:
            spatial_data = data
            
        # Accumulate
        for c in range(num_classes):
            mask = (target == c)
            if mask.sum() > 0:
                class_sums[c] += spatial_data[mask].sum(dim=0)
                class_counts[c] += mask.sum()
                
        if batch_idx % 50 == 0:
            print(f"Processed batch {batch_idx}...")

    # Calculate Means
    class_means = (class_sums / class_counts.view(-1, 1, 1)).cpu().numpy()
    
    # --- PLOTTING THE PROOF ---
    idx_A = 4 # Happy
    idx_B = 2 # Fear (or Sad)
    
    diff_map = class_means[idx_A] - class_means[idx_B]
    
    # Calculate Kurtosis of the Difference Map (The "Hotspot" Proof)
    from scipy.stats import kurtosis
    k_score = kurtosis(diff_map.flatten())
    print(f"Excess Kurtosis of Difference Map: {k_score:.4f}")
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(class_means[idx_A], cmap='viridis', cbar=False)
    plt.title(f"Mean {class_names[idx_A]}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    sns.heatmap(class_means[idx_B], cmap='viridis', cbar=False)
    plt.title(f"Mean {class_names[idx_B]}")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    # Use diverging colormap to show Positive vs Negative activation
    sns.heatmap(diff_map, cmap='seismic', center=0) 
    plt.title(f"Difference ({class_names[idx_A]} - {class_names[idx_B]})\nKurtosis={k_score:.2f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('evidence/topology_proof.png', dpi=300)
    print("Saved topology_proof.png")




def analyze_distribution(dataloader, num_batches=20, max_samples=100000):
    
    # Accumulate data
    collected_samples = []
    
    print(f"Collecting samples from {num_batches} batches...")
    for i, (data, _, _) in enumerate(dataloader):
        if i >= num_batches:
            break
        
        flat_data = data.flatten().cpu().numpy()

        if len(flat_data) > max_samples:
            flat_data = np.random.choice(flat_data, max_samples, replace=False)
            
        collected_samples.append(flat_data)

    full_distribution = np.concatenate(collected_samples)
    
    print(f"Analyzing {len(full_distribution)} data points...")

    # 1. Calculate Statistics
    mu = np.mean(full_distribution)
    sigma = np.std(full_distribution)
    
    # Fisher=True means Normal distribution has Kurtosis = 0
    kurtosis_val = stats.kurtosis(full_distribution, fisher=True)
    skewness_val = stats.skew(full_distribution)

    print(f"\n--- Statistical Audit ---")
    print(f"Mean: {mu:.4f}")
    print(f"Std Dev: {sigma:.4f}")
    print(f"Skewness: {skewness_val:.4f}")
    print(f"Excess Kurtosis: {kurtosis_val:.4f}")
    
    if kurtosis_val > 1.0:
        print(">> RESULT: Distribution is Leptokurtic (Heavy Tailed).")
    else:
        print(">> RESULT: Distribution is Mesokurtic/Platykurtic.")

    # 2. Generate Plots for Paper
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot A: Log-Scale Histogram (The best proof of fat tails)
    axes[0].hist(full_distribution, bins=100, density=True, alpha=0.7, color='blue', label='Topological Data')
    
    # Overlay Normal Distribution for comparison
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    p = stats.norm.pdf(x, mu, sigma)
    axes[0].plot(x, p, 'k', linewidth=2, label='Normal Dist')
    
    axes[0].set_yscale('log') # <--- CRITICAL: Reveals the tails
    axes[0].set_title('Log-Probability Density (Fat Tails)')
    axes[0].legend()
    axes[0].grid(True, which="both", ls="-", alpha=0.2)

    # Plot B: Q-Q Plot
    # We downsample for Q-Q plot to avoid lag
    qq_sample = np.random.choice(full_distribution, size=5000, replace=False)
    stats.probplot(qq_sample, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normality Check)')
    
    plt.tight_layout()
    plt.savefig('evidence/distribution_evidence.png', dpi=300)
    print("\nEvidence saved to 'distribution_evidence.png'")

def calculate_optimal_firing_rate(dataset, num_samples=2000):
    print("Auditing Ground Truth Energy Density...")

    total_energy = 0.0
    count = 0
    
    # Iterate through a subset of the data
    for i in range(min(len(dataset), num_samples)):
        _, target_volume, _ = dataset[i]
        
        energy_density = target_volume.sum() / target_volume.numel()
        
        total_energy += energy_density
        count += 1
        
    avg_rate = total_energy / count
    
    print(f"Measured Average Energy Density: {avg_rate:.4f}")
    print(f"Recommended target_rate: {avg_rate:.4f}")
    
    return avg_rate


def monitor_network_health(model, loader, device, epoch):
    model.eval()
    print(f"\nDIAGNOSTIC (Epoch {epoch}): Checking Network Health...")
    
    try:
        inputs, _, _ = next(iter(loader))
    except StopIteration: return

    inputs = inputs.to(device)
    if inputs.dim() == 5: inputs = inputs.permute(1, 0, 2, 3, 4)

    manual_reset(model) 
    with torch.no_grad():
        _ = model(inputs)

    print(f"\nMEMBRANE POTENTIALS:")
    print(f"{'Layer Name':<40} | {'Max':<10} | {'Mean':<10} | {'Status'}")
    print("-" * 80)

    # RECURSIVE SEARCH

    for name, module in model.named_modules():
        if isinstance(module, (snn.Leaky, snn.Synaptic, ALIF)):
            if hasattr(module, 'mem'):
                mem = module.mem
                if isinstance(mem, list): mem = mem[-1]
                
                if mem.numel() == 0: continue

                v_max = mem.max().item()
                v_mean = mem.mean().item()
                
                status = "Normal"
                if isinstance(module, ALIF):
                    if v_max > 10.0: status = "High"
                    if v_max < 0.1: status = "Dead"
                else:
                    if v_max > 5.0: status = "High"
                    if v_max < 0.05: status = "Dead"
                
                short_name = name.replace("encoder.", "Enc.").replace("decoder.", "Dec.")
                print(f"{short_name:<40} | {v_max:<10.4f} | {v_mean:<10.4f} | {status}")

    print("-" * 80)


def manual_reset(model):

    for module in model.modules():
        # Check if the module is a spiking neuron
        if isinstance(module, (snn.Leaky, snn.Synaptic, snn.Alpha)):
            if hasattr(module, 'reset_mem'):
                module.reset_mem()
            if hasattr(module, 'reset_hidden'):
                module.reset_hidden()


def apply_kaiming_init(model):
    print("Applying Kaiming (He) Initialization...")
    count = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # Fan_in preserves magnitude in the forward pass
            # Nonlinearity 'relu' is the standard proxy for SNN spikes
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            count += 1
            
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm2d, nn.InstanceNorm3d)):

            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    print(f"   Initialized {count} Convolutional layers.")
    count = 0
    for m in model.modules():
        if "GatedSkip" in m.__class__.__name__:
            # Target: m.gate -> .module -> [0] (Conv2d)
            if hasattr(m, 'gate') and hasattr(m.gate, 'module'):
                nn.init.constant_(m.gate.module[0].bias, 2.0)
                count += 1
    print(f"   Initialized {count} Open Gates.")
                
    

def run_bn_warmup(model, loader, device, num_batches=10):

    print(f"Running Data-Driven Warmup ({num_batches} batches)...")
    
    model.train()
    model.to(device)
    
    with torch.no_grad():
        for i, (inputs, _, _) in enumerate(tqdm(loader, total=num_batches, desc="Warming up")):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            if inputs.dim() == 5:
                inputs = inputs.permute(2, 0, 1, 3, 4)
            
            _ = model(inputs)
    
    print("Input BN Statistics Calibrated.")

def initialize_head(head_layer):
    # Initialize weights with very small variance to keep logits similar
    nn.init.normal_(head_layer.module.weight, mean=0.0, std=0.01)
    
    # Set bias to 0 so no class has an advantage
    if head_layer.module.bias is not None:
        nn.init.constant_(head_layer.module.bias, 0)

def initialize_vit(vit_layer):
    print("Initializing ViT Heads...")
    if isinstance(vit_layer.mlp, SwiGLU):
        nn.init.zeros_(vit_layer.mlp.out.weight)
        if vit_layer.mlp.out.bias is not None:
            nn.init.zeros_(vit_layer.mlp.out.bias)
    else:
        nn.init.zeros_(vit_layer.mlp[-1].weight)
        nn.init.zeros_(vit_layer.mlp[-1].bias)

    nn.init.zeros_(vit_layer.attn.out_proj.weight)
    nn.init.zeros_(vit_layer.attn.out_proj.bias)

def initialize_reconv(reconv_layer):
    print("Initializing ReConv Layer...")
    nn.init.zeros_(reconv_layer.module.weight)

def initialize_network(model, train_loader, device):

    apply_kaiming_init(model)
    run_bn_warmup(model, train_loader, device)
    initialize_head(model.classifier.head)
    if model.encoder.vit:
        initialize_vit(model.encoder.temporal_vit)
    if hasattr(model, 'decoder') and model.decoder.reccurent:
        initialize_reconv(model.decoder.up1.conv1.spike.recurrent_conv)
        initialize_reconv(model.decoder.up1.conv2.spike.recurrent_conv)

def calculate_p98(dataset):

    sz = int(0.05 * len(dataset))
    print(f"   Sampling {sz} random files for statistics...")
    reservoir = []

    for i in tqdm(range(sz), desc="Calculating P98"):
        video, _, _ = dataset[i]
        pixels = video.flatten().numpy()
        active = pixels[pixels > 1e-9]

        if len(active) > 0:
            choice = np.random.choice(active, size=min(500, len(active)), replace=False)
            reservoir.append(choice)

    all_samples = np.concatenate(reservoir)
    p98 = np.percentile(all_samples, 98)
    print(f"Calculated 98th Percentile of Active Pixels: {p98:.6f} on reservoir of {len(all_samples)} samples.")

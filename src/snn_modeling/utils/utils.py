import torch
import torch.nn as nn
from tqdm import tqdm
import snntorch as snn
from ..layers.neurons import ALIF

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
            
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):

            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    print(f"   Initialized {count} Convolutional layers.")

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

def initialize_network(model, train_loader, device):

    apply_kaiming_init(model)
    run_bn_warmup(model, train_loader, device)
    initialize_head(model.classifier.head)
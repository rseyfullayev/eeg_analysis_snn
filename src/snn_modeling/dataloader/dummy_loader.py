import torch
import numpy as np

def make_gaussian(size, center=None, sigma=10):
    """Generates a 2D Gaussian blob."""
    x = torch.arange(0, size, 1).float()
    y = torch.arange(0, size, 1).float()
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    
    if center is None:
        center = (size // 2, size // 2)
    
    x0, y0 = center
    return torch.exp(-((x_grid - x0)**2 + (y_grid - y0)**2) / (2 * sigma**2))
def get_dummy_batch(config, device):
    
    batch_size = config['training']['batch_size']
    in_channels  = config['model']['in_channels']
    num_steps = config['data']['num_timesteps']
    height = 64
    width = 64
    num_classes = config['model']['num_classes']
    print(f"Creating dummy data: Input {num_steps}x{batch_size}x{in_channels}x{height}x{width}")

    inputs = torch.rand(num_steps, batch_size, in_channels, height, width).to(device)
    inputs = (inputs > 0.9).float() 

    targets = torch.zeros(batch_size, num_classes, height, width).to(device)
    targets_c = torch.randint(0, num_classes-1, (batch_size,), dtype=torch.long).to(device)
    blob = torch.rand(batch_size, num_classes, height, width).to(device) #make_gaussian(height, sigma=config['mask']['sigma']).to(device)
    blob = blob / blob.max()
    #for b in range(batch_size):

        #targets[b, targets_c[b], :, :] = blob
        
    return inputs, targets, targets_c
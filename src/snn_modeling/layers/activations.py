import torch.nn as nn
import snntorch as snn
import inspect
from .neurons import ALIF

ACTIVATION_MAP = {
    "ALIF": ALIF,
    "Leaky": snn.Leaky,
    "Synaptic": snn.Synaptic,
    "Alpha": snn.Alpha,
    "SiLU": nn.SiLU,
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "Identity": nn.Identity,
    # Legacy mappings
    "snn.Leaky": snn.Leaky,
    "snn.Synaptic": snn.Synaptic,
    "snn.Alpha": snn.Alpha,
    "nn.SiLU": nn.SiLU,
}

def resolve_activation(spike_model):
    """Resolves string class names to their actual classes."""
    if isinstance(spike_model, str):
        return ACTIVATION_MAP.get(spike_model, nn.Identity)
    return spike_model

def is_alif(spike_model):
    """Safe check if the model is ALIF."""
    # If passed as a string
    if isinstance(spike_model, str):
        return spike_model == 'ALIF'
    # If passed as a class
    return getattr(spike_model, '__name__', '') == 'ALIF'

def instantiate_activation(spike_model, **kwargs):
    """Safely instantiates an activation, filtering kwargs if necessary."""
    spike_model_class = resolve_activation(spike_model)
    
    # Fast paths for simple PyTorch activations that take no arguments by default
    if spike_model_class in (nn.SiLU, nn.ReLU, nn.Identity, nn.GELU):
        return spike_model_class()
        
    # For models that take parameters, filter out kwargs they don't accept
    sig = inspect.signature(spike_model_class.__init__)
    valid_keys = sig.parameters.keys()
    
    filtered_kwargs = {
        k: v for k, v in kwargs.items() 
        if k in valid_keys or k == 'kwargs' or any(
            param.kind == inspect.Parameter.VAR_KEYWORD 
            for param in sig.parameters.values()
        )
    }
    
    return spike_model_class(**filtered_kwargs)

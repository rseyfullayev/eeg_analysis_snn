
import snntorch as snn
from ..models.unet import SpikingUNet, UNet
from ..models.encoders import SpikingMobileNetEncoder
from hydra.utils import instantiate
import inspect
from ..layers.activations import ACTIVATION_MAP

def get_filtered_neuron_params(neuron_class, full_config_params):
    """
    Inspects the neuron_class (e.g., snn.Leaky) and filters the full_config_params
    to return only the arguments that this specific neuron actually accepts.
    """
   
    sig = inspect.signature(neuron_class.__init__)
    valid_keys = sig.parameters.keys()

    filtered_params = {
        k: v for k, v in full_config_params.items() 
        if k in valid_keys
    }

    return filtered_params


def build_model(config):
    # 1. Native Hydra Instantiation! (The clean way)
    if hasattr(config.model, '_target_'):
        return instantiate(config.model, _recursive_=True)

    # 2. Legacy / Fallback manually building the model 
    model_type = config.model.type
    
    if model_type == "SpikingUNet":
        if config.model.encoder_type == "MobileNet":
            encoder = SpikingMobileNetEncoder
        else:
            raise ValueError(f"Unknown encoder type: {config.model.encoder_type}")
        spike_model_class = ACTIVATION_MAP[config.neuron_params.spike_model]
        snn_params = {
            'alpha': config.neuron_params.get('alpha', 0.5),
            'beta': config.neuron_params.get('beta', 0.9),
            'threshold': config.neuron_params.get('threshold', 1.),
            'gamma_adapt': config.neuron_params.get('gamma_adapt', 0.5),
            'decay_adapt': config.neuron_params.get('decay_adapt', 0.96),
            'learn_alpha': config.neuron_params.get('learn_alpha', False),
            'learn_beta': config.neuron_params.get('learn_beta', False),
            'learn_decay': config.neuron_params.get('learn_decay', False),
            'learn_gamma': config.neuron_params.get('learn_gamma', False),
            'learn_threshold': config.neuron_params.get('learn_threshold', False),
            'learn_slope': config.neuron_params.get('learn_slope', False),
            'reset_mechanism': config.neuron_params.get('reset_mechanism', 'zero')
        }
        actual_params = get_filtered_neuron_params(spike_model_class, snn_params)
       
        model = SpikingUNet(
            config=config,
            encoder=encoder,
            in_channels=config.model.in_channels,
            num_classes=config.model.num_classes,
            spike_model=spike_model_class,
            **actual_params
        )
        
    elif model_type == "UNet":
        if config.model.encoder_type == "ResNet18":
            #encoder = ResNet18Encoder
            pass
        else:
            raise ValueError(f"Unknown encoder type: {config.model.encoder_type}")
        spike_model_class = ACTIVATION_MAP[config.neuron_params.spike_model]

        model = UNet(
            encoder=encoder,
            in_channels=config.model.in_channels,
            num_classes=config.model.num_classes
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
       
    return model
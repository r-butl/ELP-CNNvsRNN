"""
PyTorch MobileNetV2 model for binary classification
Compatible with Apple Silicon and supports QAT and PTQ
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.quantization import QuantStub, DeQuantStub


class MobileNetV2Model(nn.Module):
    """MobileNetV2 with custom classifier head"""
    
    def __init__(self, model_config, training=True, input_shape=None, pretrained=False):
        super().__init__()
        
        # Determine if grayscale input
        self.grayscale_input = (input_shape is not None and input_shape[-1] == 1)
        
        # Load base MobileNetV2 model (small variant for efficiency)
        self.base_model = models.mobilenet_v2(pretrained=pretrained)
        
        # Grayscale to RGB conversion if needed
        if self.grayscale_input:
            self.rgb_conv = nn.Conv2d(1, 3, kernel_size=1)
        else:
            self.rgb_conv = None
        
        # Get activation function
        if model_config['activation_function'] == 'ReLU':
            activation_fn = nn.ReLU
        elif model_config['activation_function'] == 'LeakyReLU':
            activation_fn = nn.LeakyReLU
        else:
            activation_fn = nn.ReLU
        
        # Replace classifier
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 256),
            activation_fn(),
            nn.Dropout(p=model_config['dropout_rate']),
            nn.Linear(256, 50),
            activation_fn(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )
        
        # Set training mode
        self.base_model.train(training)
        
        # Quantization stubs (needed for QAT)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        
        if self.rgb_conv is not None:
            x = self.rgb_conv(x)
        
        x = self.base_model(x)
        x = self.dequant(x)
        return x


class QuantizedMobileNetV2Model(nn.Module):
    """MobileNetV2 with quantization-aware training support"""
    
    def __init__(self, model_config, training=True, input_shape=None, pretrained=False):
        super().__init__()
        
        # Create base model
        self.model = MobileNetV2Model(
            model_config=model_config,
            training=training,
            input_shape=input_shape,
            pretrained=pretrained
        )
        
        # Configure for QAT if training
        if training:
            self.model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    
    def forward(self, x):
        return self.model(x)
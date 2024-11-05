import torch
import torch.nn as nn
import torch.nn.functional as F
from loralib.layers import Linear

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.lora_layer = Linear(20, 20, r=4, lora_alpha=1, lora_dropout=0.1, merge_weights=True)

    def forward(self, x):
        return self.lora_layer(x)

def test_lora_layer():
    model = TestModel()
    x = torch.randn(10, 20)
    output = model(x)
    assert output.shape == (10, 20), "Output shape mismatch"
    
    # Check if gradients flow correctly
    output.sum().backward()
    for name, param in model.named_parameters():
        if 'lora_' in name:
            assert param.grad is not None, "Gradient not flowing"
            assert param.grad is not None, "Gradient not flowing"

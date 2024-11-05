import torch
import torch.nn as nn
import torch.nn.functional as F
from loralib.layers import LoRALayer

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.lora_layer = LoRALayer(r=4, lora_alpha=1, lora_dropout=0.1, merge_weights=True)

    def forward(self, x):
        return self.lora_layer(x)

def test_lora_layer():
    model = TestModel()
    x = torch.randn(10, 20)
    output = model(x)
    assert output.shape == (10, 20), "Output shape mismatch"
    
    # Check if gradients flow correctly
    output.sum().backward()
    for param in model.parameters():
        assert param.grad is not None, "Gradient not flowing"

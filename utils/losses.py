import torch
import torch.nn as nn

class MultipleMSELoss(nn.Module):
    def __init__(self):
        super(MultipleMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        final_loss = 0
        for output in outputs:
            final_loss += self.mse_loss(output, targets)
        return final_loss / len(outputs)
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegularizedLoss(nn.Module):
    def __init__(self, model, lambda_reg):
        super(RegularizedLoss, self).__init__()
        self.model = model
        self.lambda_reg = lambda_reg

    def forward(self, device, inputs, targets):
        outputs = self.model(inputs)
        criterion = nn.MSELoss()
        loss = criterion(outputs, targets)

        l2_reg = torch.tensor(0.0).to(device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param.to(device))

        loss += self.lambda_reg * l2_reg

        return loss
from packages.predictor.net import Predictor
from packages.predictor_trainer.regularized_loss import RegularizedLoss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Predictor(
        interval_num=4, 
        predictor_hidden_dim=16, 
        section_input_dim=3, 
        section_hidden_1_dim=32,
        section_hidden_2_dim=8
    ).to(device)

    criterion = RegularizedLoss(model, lambda_reg=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3)
    
    input_data = torch.tensor(
        [
            [
                [0, 0, 1, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ]
        ], 
        dtype=torch.float32
    ).to(device)
    target = torch.tensor(
        [
            [2.85]
        ], 
        dtype=torch.float32
    ).to(device)
    print(input_data.shape)
    print(target.shape)

    num_epochs = 200

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # output = model(input_data)
        loss = criterion(device, input_data, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 1:
            print(f'Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item():.4f}')
    

    with torch.no_grad():
        out_tensor = model(input_data)
    sr_predicted = out_tensor.detach().cpu().numpy()[0, 0]
    print(sr_predicted)
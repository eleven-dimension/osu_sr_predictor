from packages.predictor.net import Predictor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    p = Predictor(
        section_num=4, 
        predictor_hidden_dim=16, 
        section_input_dim=3, 
        section_hidden_1_dim=32,
        section_hidden_2_dim=8
    )

    out = p(torch.rand(6, 3, 4))
    print(out.shape)
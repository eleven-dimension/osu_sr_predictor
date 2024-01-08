import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SectionBlock(nn.Module):
    def __init__(
            self,
            input_dim = 4 * 2 * 100,
            hidden_1_dim = 256,
            hidden_2_dim = 16
    ):
        super(SectionBlock, self).__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_1_dim)
        self.fc_2 = nn.Linear(hidden_1_dim, hidden_2_dim)
        self.section_val_out = nn.Linear(hidden_2_dim, 1)


    # state shape [N, 8, 100]
    def forward(self, state):
        x = F.relu(self.fc_1(state))
        x = F.relu(self.fc_2(x))
        val = self.section_val_out(x)

        return val
    

class Predictor(nn.Module):
    def __init__(
            self,
            interval_num = 1000,
            predictor_hidden_dim = 256,

            section_input_dim = 4 * 2 * 100,
            section_hidden_1_dim = 256,
            section_hidden_2_dim = 16
    ):
        super(Predictor, self).__init__()
        self.section_block = SectionBlock(
            section_input_dim, section_hidden_1_dim, section_hidden_2_dim
        )
        self.fc_1 = nn.Linear(interval_num, predictor_hidden_dim)
        self.val = nn.Linear(predictor_hidden_dim, 1)


    # state shape [N, 800, 1000] [batch_n, dim_num, interval_num]
    # out: [N, 1]
    def forward(self, state):
        split_tensors = torch.split(state, split_size_or_sections=1, dim=-1)

        section_outputs = [None for _ in range(state.shape[-1])]
        for i, split_tensor in enumerate(split_tensors):
            section_i_input = split_tensor.view(split_tensor.shape[0], split_tensor.shape[1]) # [N, 800]
            section_outputs[i] = self.section_block(section_i_input) # [N, 1]
        
        fc_input = torch.cat(section_outputs, dim=1)
        x = F.relu(self.fc_1(fc_input))
        val = self.val(x)

        return val
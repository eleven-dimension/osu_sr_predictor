import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    x = torch.tensor(
        [
            [
                [1, 4, 7, -1], 
                [2, 5, 8, -1], 
                [3, 6, 9, -1]
            ],
            [
                [9, 7, 7, -4], 
                [9, 5, 7, -3], 
                [9, 1, 1, -2]
            ]
        ]
    )

    print(x.shape) # [2, 3, 4]

    split_tensors = torch.split(x, split_size_or_sections=1, dim=-1)
    for i, split_tensor in enumerate(split_tensors):
        print(f"Tensor {i + 1}: {split_tensor.squeeze().shape}")
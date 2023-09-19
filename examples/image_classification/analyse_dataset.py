#OFL File
import torch
import numpy as np

def dataset_mean(dataset):
    l = [data[0] for i, data in enumerate(dataset)]
    l = tuple(l)
    print("Type:", type(l[0]))
    stacked_tensor = torch.stack(l)
    # print("Dataset:", stacked_tensor)
    calculated_mean = torch.mean(stacked_tensor)
    return calculated_mean
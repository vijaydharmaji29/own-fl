#OFL File
import torch
import torch.nn as nn
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


class Analyser:
    
    def __init__(self):
        print("\n\tInitialised Analyser Object\t\n")

    def calculate_dataset_mean(self, dataset_tensor):
        l = [data[0] for i, data in enumerate(dataset_tensor)]
        l = tuple(l)
        stacked_tensor = torch.stack(l)
        calculated_mean = torch.mean(stacked_tensor)
        return calculated_mean

    def calculate_dataset_std(self, dataset_tensor):
        l = [data[0] for i, data in enumerate(dataset_tensor)]
        l = tuple(l)
        stacked_tensor = torch.stack(l)
        calculated_std = torch.std(stacked_tensor)
        return calculated_std

    def cosine_similarity(self, tensor1, tensor2):
        cl = []

        cos = nn.CosineSimilarity(dim=0)

        for t in tensor1:
            cl.append(torch.norm(torch.flatten(cos(t, tensor2))))

        return cl
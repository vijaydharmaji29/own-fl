import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

class DataStorage:
    def __init__(self):
        print("\n\tInitialised DataStorage object\n\t")
        self.global_accuracy = []
        self.train_dataset_tensors_list = []
        self.train_dataset_tensors_labels_list = []
        self.local_round_loss = []
        self.dataset_means = []

    def add_global_accuracy(self, ga):
        self.global_accuracy.append(ga)
    
    def add_local_round_loss(self, lrl):
        self.local_round_loss.append(lrl)

    def add_train_dataset_tensors(self, tdt):
        d = {}
        
        tensor_l = []

        for i, data in enumerate(tdt):
            tensor_l.append(data[0][0])
            self.train_dataset_tensors_labels_list.append(data[1][0])

            if data[1][0].item() in d:
                d[data[1][0].item()] += 1
            else:
                d[data[1][0].item()] = 1

        tensor_l = tuple(tensor_l)


        stacked_tensor = torch.stack(tensor_l)

        self.train_dataset_tensors_list.append(stacked_tensor)
        

        print("Here is the dictionary, yippie:", d)

    def get_train_dataset_tensors_list(self):
        return self.train_dataset_tensors_list
    
    def add_dataset_mean(self, mean):
        self.dataset_means.append(mean)
    
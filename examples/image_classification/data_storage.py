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
        self.label_distribution = []

    def add_global_accuracy(self, ga):
        self.global_accuracy.append(ga)
    
    def add_local_round_loss(self, lrl):
        self.local_round_loss.append(lrl)

    def add_train_dataset_tensors(self, tdt):
        d = {}
        
        tensor_l = []
        label_list = []

        for i, data in enumerate(tdt):
            tensor_l.append(data[0][0])
            label_list.append(data[1][0].item())

            if data[1][0].item() in d:
                d[data[1][0].item()] += 1
            else:
                d[data[1][0].item()] = 1

        #ordering tensor_l based on corresponding labels
        for k in range(len(label_list)-1):
            if label_list[k] > label_list[k+1]:
                label_list[k], label_list[k+1] = label_list[k+1], label_list[k]
                tensor_l[k], tensor_l[k+1] = tensor_l[k+1], tensor_l[k]


        tensor_l = tuple(tensor_l)


        stacked_tensor = torch.stack(tensor_l)

        self.train_dataset_tensors_list.append(stacked_tensor)
        self.train_dataset_tensors_labels_list.append(label_list)
        self.label_distribution.append(d)
        
    def get_train_dataset_tensors_list(self):
        return self.train_dataset_tensors_list
    
    def add_dataset_mean(self, mean):
        self.dataset_means.append(mean)
    
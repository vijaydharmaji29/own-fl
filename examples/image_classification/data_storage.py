import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

class DataStorage:
    def __init__(self):
        print("\n\tInitialised DataStorage object\n\t")
        self.global_accuracy = [] #accuracy of global model after each round
        self.train_dataset_tensors_list = [] #list of dataset tensors of each round
        self.train_dataset_tensors_labels_list = [] #list of label list of each round
        self.local_round_loss = [] #local round loss of each round
        self.dataset_means = [] #calculated dataset means of each calculated round
        self.label_distribution = [] #number of each label of each round
        self.label_distribution_similarity_list = [] #similarity of label distribution of each round with their previous rounds(nxn matrix -> in essence)
        self.cossim_list = [] #similarity of dataset distribution of each round with their previous rounds (nxn matrix -> in essence)

    def add_global_accuracy(self, ga):
        self.global_accuracy.append(ga)
    
    def add_local_round_loss(self, lrl):
        self.local_round_loss.append(lrl)


    #calculates dot product of 2 lists of numbers
    def dot_product(self, l1, l2):
        if len(l1) != len(l2):
            return None
        else:
            d = 0

            for i in range(len(l1)):
                d += l1[i]*l2[i]

            return d

    def add_train_dataset_tensors(self, tdt):
        d = {}
        
        tensor_l = []
        label_list = []

        #getting images and their corresponding labels from the datset recieved
        for i, data in enumerate(tdt):
            tensor_l.append(data[0][0])
            label_list.append(data[1][0].item())

            if data[1][0].item() in d:
                d[data[1][0].item()] += 1
            else:
                d[data[1][0].item()] = 1

        #ordering tensor_l (list of tensors of images) based on corresponding labels
        for k in range(len(label_list)-1):
            if label_list[k] > label_list[k+1]:
                label_list[k], label_list[k+1] = label_list[k+1], label_list[k]
                tensor_l[k], tensor_l[k+1] = tensor_l[k+1], tensor_l[k]


        tensor_l = tuple(tensor_l)

        stacked_tensor = torch.stack(tensor_l) #storing all images recieved into one tensor

        self.train_dataset_tensors_list.append(stacked_tensor)
        self.train_dataset_tensors_labels_list.append(label_list)

        d_l = [0]*len(d)

        for key in d:
            d_l[int(key)] = d[key]

        self.label_distribution.append(d_l)
        
        label_similarity = [] #similarity list for current label distribution with all past label distributions

        #calcualting dot product of current label distribution and past label distributions
        for l1 in self.label_distribution:
            print(l1)
            print(d_l)
            dot_product_sim = self.dot_product(l1, d_l)
            if dot_product_sim:
                label_similarity.append(dot_product_sim) #appends dot product of current label distribution and ith label distribution 
            else:
                label_similarity.append(0)
                print("ERROR IN LABEL DISTRIBUTION SIMILARITY")

        self.label_distribution_similarity_list.append(label_similarity) #appending the similarities for current label distribution with past label distributions to the main list

    def get_train_dataset_tensors_list(self):
        return self.train_dataset_tensors_list
    
    def add_dataset_mean(self, mean):
        self.dataset_means.append(mean)

    def add_cosinesimilarity(self, sim):
        self.cossim_list.append(sim)
    
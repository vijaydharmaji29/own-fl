from typing import Tuple
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

class DataManger:
    """
    Managing training/test data
    Note: Singleton Pattern - commented out for now
    """
    # _singleton_dm = None

    # @classmethod
    # def dm(cls, th: int = 0):
    #     if not cls._singleton_dm and th > 0:
    #         cls._singleton_dm = cls(th)
    #     return cls._singleton_dm

    def __init__(self, cutoff_th: int):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=None)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        
            
        #OFL - to access the trainset variable above
        #accessed in execute_ic_training method
        N = int(len(trainset))
        # generate & shuffle indices
        indices = np.arange(N)
        indices = np.random.permutation(indices)
        train_indices = indices[:int(N*0.1)]

        self.pt = torch.utils.data.Subset(trainset, train_indices)
        self.public_trainset_PIL = MyDataset(self.pt)
        self.public_trainset = MyDataset(self.pt, transform=transform) #using custom dataset to transform PILImages to tensors for training


        self.trainloader = torch.utils.data.DataLoader(self.public_trainset, batch_size=4,
                                                       shuffle=True, num_workers=2)

        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                      shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')

        self.cutoff_threshold = int(cutoff_th*0.1)


    def get_random_images(self, is_train: bool = False) -> Tuple:
        """
        Retrun a batch of images and labels
        Those can be used to show examples for demos
        :param is_train:
        :return:
        """
        if is_train:  # if it requires training data
            ldr = self.trainloader
        else:  # test data
            ldr = self.testloader
        imgs, labels = iter(ldr).next()

        return imgs, labels


def execute_ic_training(dm, net, criterion, optimizer):
    """
    Training routine
    :param dm: DataManager providing access to training data
    :param net: CNN
    :param criterion:
    :param optimizer:
    :return:
    """
    # To simulate the scenarios where each agent has less number of data
    # it exists from training after iterating till the cutoff threshold
    random_indices = random.sample(range(0, len(dm.trainloader)), dm.cutoff_threshold)

    final_running_loss = 0
    for epoch in range(1):
        running_loss = 0.0
        j = 0
        for i, data in enumerate(dm.trainloader):
            if i in random_indices:
                j += 1
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                final_running_loss = running_loss
                if j % 1000 == 999:  # print every 1000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, j + 1, running_loss / 1000))
                    # print("Loss.item: ", loss.item())
                    running_loss = 0.0

    #OFL
    print("Final running loss for round =", final_running_loss)
    print("Trainset type: ", type(dm.trainloader))
    return net, final_running_loss, dm.trainloader, dm.public_trainset_PIL



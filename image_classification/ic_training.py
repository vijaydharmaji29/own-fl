from typing import Tuple
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import math

torch.multiprocessing.set_sharing_strategy('file_system')


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

    def __init__(self, cutoff_th: int, last_accuracy=[1] * 10, order=True):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=None)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

        # OFL - to access the trainset variable above
        # accessed in execute_ic_training method
        # generate skewed indices for trainset label distribution

        N = int(len(trainset))
        shuffled_indices = np.arange(N)
        shuffled_indices = np.random.permutation(shuffled_indices)

        n_labels = 10
        order_labels = np.arange(n_labels)
        order_labels = np.random.permutation(
            order_labels)  # this is the order of distributions, comment it out to keep the same label skewed
        order_labels = np.array(order_labels)

        train_indices = []
        max_indices = N * .1

        # adding to indices:
        label_added_count = [0] * 10
        max_count_per_label = {}

        # adding max count per label
        for i in range(len(order_labels)):
            max_count_per_label[order_labels[i]] = ((i + 1) * max_indices / 55)

        # adding the required images for training
        for i in shuffled_indices:
            current_label = trainset[i][1]
            try:
                if label_added_count[current_label] < max_count_per_label[current_label]:
                    train_indices.append(i)
                    label_added_count[current_label] += 1
            except:
                # pass
                print("ERROR")
                print(current_label)
                print(label_added_count)
                print(max_count_per_label)

        # train_indices = shuffled_indices[:int(N*.1)] #for random distribution

        training_label_distribution = [0] * n_labels
        for i in max_count_per_label:
            training_label_distribution[i] = max_count_per_label[i]

        # self.dataset_similarity_score = similarity_score(training_label_distribution, last_accuracy)
        self.dataset_bhattacharya_distance = bhattacharya_distance(training_label_distribution, last_accuracy)

        self.pt = torch.utils.data.Subset(trainset, train_indices)
        self.public_trainset_PIL = MyDataset(self.pt)
        self.public_trainset = MyDataset(self.pt,
                                         transform=transform)  # using custom dataset to transform PILImages to tensors for training

        self.trainloader = torch.utils.data.DataLoader(self.public_trainset, batch_size=1,
                                                       shuffle=True, num_workers=1)

        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                      shuffle=True, num_workers=1)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')

        self.cutoff_threshold = int(cutoff_th * 0.1)

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


# converting a distribution to standard normal form
def convert_to_snf(l):
    mean = 0
    N = len(l)

    for i in l:
        mean += i

    mean = mean / N

    sd = 0

    for i in l:
        sd += (i - mean) * (i - mean)

    sd = sd / N

    if sd == 0:
        return l

    sd = math.sqrt(sd)

    new_l = []

    for x in l:
        new_x = (x - mean) / sd
        new_l.append(new_x)

    return new_l


def similarity_score(l1, l2):  # lower the better
    l1 = convert_to_snf(l1)
    l2 = convert_to_snf(l2)

    if (len(l1) != len(l2)):
        return -1

    score = 0

    for i in range(len(l1)):
        score += (l1[i] - l2[i]) * (l1[i] - l2[i])

    return score


def bhattacharya_distance(l1, l2):
    # getting pdf
    l1 = [x / sum(l1) for x in l1]
    l2 = [x / sum(l2) for x in l2]

    bc = 0

    for i in range(len(l1)):
        bc += math.sqrt(l1[i] * l2[i])

    bd = -math.log(bc)

    return bd


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
    # random_indices = random.sample(range(0, len(dm.trainloader)), len(dm.trainloader))

    label_distribution = [0] * 10

    final_running_loss = 0
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(dm.trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            label_distribution[labels[0]] += 1

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

    # OFL
    print("Label distribution: ", label_distribution)
    print("Final running loss for round =", final_running_loss)
    return net, final_running_loss, dm.trainloader, dm.public_trainset_PIL
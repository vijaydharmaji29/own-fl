import logging
from typing import Dict
import csv
import pandas as pd
import sys


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .cnn import Net
from .conversion import Converter
from .ic_training import DataManger, execute_ic_training
from .analyse_dataset import Analyser
from .data_storage import DataStorage as ds

from fl_main.agent.client import Client

torch.multiprocessing.set_sharing_strategy('file_system')

name = ""

class TrainingMetaData:
    # The number of training data used for each round
    # This will be used for the weighted averaging
    # Set to a natural number > 0
    num_training_data = 8000

def init_models() -> Dict[str,np.array]:
    """
    Return the templates of models (in a dict) to tell the structure
    The models need not to be trained
    :return: Dict[str,np.array]
    """
    net = Net()
    return Converter.cvtr().convert_nn_to_dict_nparray(net)

def training(models: Dict[str,np.array], init_flag: bool = False, DataStorage = None, similarity_score_treshold = 17, order = True) -> Dict[str,np.array]:
    """
    A place holder function for each ML application
    Return the trained models
    Note that each models should be decomposed into numpy arrays
    Logic should be in the form: models -- training --> new local models
    :param models: Dict[str,np.array]
    :param init_flag: bool - True if it's at the init step.
    False if it's an actual training step
    :return: Dict[str,np.array] - trained models
    """
    # return templates of models to tell the structure
    # This model is not necessarily actually trained
    if init_flag:
        # Prepare the training data
        # num of samples / 4 = threshold for training due to the batch size

        # dm = DataManger(int(TrainingMetaData.num_training_data / 4))
        return init_models()

    # Do ML Training
    logging.info(f'--- Training ---')

    # Create a CNN based on global (cluster) models
    net = Converter.cvtr().convert_dict_nparray_to_nn(models)

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # models -- training --> new local models
    last_accuracy = [1] * 10
    if len(DataStorage.label_accuracy) > 0:
        last_accuracy = DataStorage.label_accuracy[-1]

    dm = DataManger(int(TrainingMetaData.num_training_data / 4), last_accuracy, order)
    print("ds label accuracy:", last_accuracy)
    data_object_for_training = dm #instance of DataManager object

    #returns trained neural network,
    #dataset used for training in terms of tensor (after tranformation)
    #dataset used for trainign in terms of PIL (before transformation)

    similarity_score = dm.dataset_similarity_score

    # test for similarity score
    print('\n\n\n')
    print("Similarity Score: ", similarity_score)
    print('\n\n\n')

    if similarity_score <= similarity_score_treshold:
        return models, None, None, None

    trained_net, round_loss, train_dataset_tensors, trainset_dataset_PIL = execute_ic_training(data_object_for_training, net, criterion, optimizer)
    
    models = Converter.cvtr().convert_nn_to_dict_nparray(trained_net)
    return models, round_loss, train_dataset_tensors, trainset_dataset_PIL

def compute_performance(models: Dict[str,np.array], testdata, is_local: bool) -> float:
    """
    Given a set of models and test dataset, compute the performance of the models
    :param models:
    :param testdata:
    :return:
    """
    # Convert np arrays to a CNN
    net = Converter.cvtr().convert_dict_nparray_to_nn(models)

    correct = 0
    total = 0
    dm = DataManger(int(TrainingMetaData.num_training_data / 4))

    labels_predicted_correctly = [0]*10
    labels_total = [0]*10

    with torch.no_grad():
        for data in dm.testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            for i in range(len(labels)):
                labels_total[labels[i]] += 1
                if labels[i] == predicted[i]:
                    labels_predicted_correctly[labels[i]] += 1
            
            correct += (predicted == labels).sum().item()

    labels_accuracy = []

    for i in range(len(labels_predicted_correctly)):
        labels_accuracy.append(labels_predicted_correctly[i]/labels_total[i])
        
    print("Label wise accuracy", labels_accuracy)

    acc = float(correct) / total

    mt = 'local'
    if not is_local:
        mt = 'Global'

    print(f'Accuracy of the {mt} model with the 10000 test images: {100 * acc} %%')

    return acc, labels_accuracy

def judge_termination(training_count: int = 0, gm_arrival_count: int = 0, training_count_treshold: int = 20) -> bool:

    if training_count >= training_count_treshold:
        return False
    """
    Decide if it finishes training process and exits from FL platform
    :param training_count: int - the number of training done
    :param gm_arrival_count: int - the number of times it received global models
    :return: bool - True if it continues the training loop; False if it stops
    """

    # could call a performance tracker to check if the current models satisfy the required performance
    return True

def prep_test_data():
    testdata = 0
    return testdata

def write_analysis(DataStorage):
    # global_accuracies = DataStorage.get_global_accuracies()
    # local_accuracies = DataStorage.get_local_accuracies()
    # local_round_loss = DataStorage.get_local_round_loss()
    # dataset_means = DataStorage.get_dataset_means()
    # dataset_stds = DataStorage.get_dataset_stds()
    # dataset_tensor_similarities = DataStorage.get_dataset_tensor_similarities()
    # label_distribution_similarities = DataStorage.get_label_distribution_similarities()

    local_accuracies = DataStorage.get_local_accuracies()

    print("\n\nPRINTING ANALYSIS\n\n")
    print(local_accuracies)
    print(DataStorage.get_global_accuracies())

    print('\n\nLABEL ACCURACIES\n\n')
    labels_accuracy = DataStorage.label_accuracy

    df = pd.DataFrame(labels_accuracy)
    df.to_csv("./test_files/label_accuracy_" + name + ".csv")

    df1 = pd.DataFrame(DataStorage.global_label_accuracy)
    df1.to_csv("./test_files/global_label_accuracy.csv")

    print("\n\nGLOBAL LABEL ACCURACIES\n\n")
    print(DataStorage.global_label_accuracy)
    print("\n\nDONE\n\n")


    print("\n\nDONE\n\n")



if __name__ == '__main__':

    #to check if reverse skewing is enabled
    try:
        name = sys.argv[3]
        order = sys.argv[4]
        if order == "reverse":
            order = False
            print("Reverse is enabled")
        else:
            order = True
    except:
        order = True

    print("Order = ", order)

    logging.basicConfig(level=logging.INFO)
    logging.info('--- Heterogeneity Aware FL with client level intelligenece ---')

    fl_client = Client()
    logging.info(f'--- Your IP is {fl_client.agent_ip} ---')

    # Create a set of template models (to tell the shapes)
    initial_models = training(dict(), init_flag=True, order=order)

    # Sending initial models
    fl_client.send_initial_model(initial_models)

    # Starting FL client
    fl_client.start_fl_client()

    training_count = 0
    gm_arrival_count = 0

    #number of rounds of training to run
    training_count_treshold = 20
    
    DataStorage = ds()
    AD = Analyser()

    while judge_termination(training_count, gm_arrival_count, training_count_treshold):

        print("\n\t TRAINING COUNT: ", training_count, "\n")

        # Wait for Global models (base models)
        global_models = fl_client.wait_for_global_model()
        gm_arrival_count += 1

        # Global Model evaluation (id, accuracy)
        global_model_performance_data = compute_performance(global_models, prep_test_data(), False)
        #add model to add global model performance data to database

        DataStorage.add_global_label_accuracy(global_model_performance_data[1])

        DataStorage.add_global_accuracy(global_model_performance_data[0])#this is lagged by one 

        # Training
        models, round_loss, train_dataset_tensors, trainset_dataset_PIL = training(global_models, DataStorage=DataStorage, order=order, similarity_score_treshold=20)
        
        if(round_loss == None): #condition for checking if client is not participating
            # Sending initial models
            logging.info(f'--- FAILED SIMILARITY SCORE TRESHOLD, SKIPPING CURRENT ROUND ---')
            fl_client.send_initial_model(global_models, num_samples=0, perf_val=-1) #sending sample if similarity score treshold failed
            continue
        
        training_count += 1
        logging.info(f'--- Training Done ---')

        # Local Model evaluation (id, accuracy)
        accuracy, labels_accuracy = compute_performance(models, prep_test_data(), True)
        DataStorage.add_label_accuracy(labels_accuracy)
        DataStorage.add_local_accuracy(accuracy)
        fl_client.send_trained_model(models, int(TrainingMetaData.num_training_data), accuracy)

    write_analysis(DataStorage)

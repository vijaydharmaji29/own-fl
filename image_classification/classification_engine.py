import logging
from typing import Dict
import csv
import pandas as pd
import sys
import datetime
import threading
import os
import random

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
from .system_measurement import SystemMeasurement

torch.multiprocessing.set_sharing_strategy('file_system')


class TrainingMetaData:
    # The number of training data used for each round
    # This will be used for the weighted averaging
    # Set to a natural number > 0
    num_training_data = 8000


def select_item_with_probability(items, probabilities):
    if len(items) != len(probabilities):
        raise ValueError("The number of items and probabilities should be the same.")

    selected_item = random.choices(items, weights=probabilities, k=1)
    return selected_item[0]


def init_models() -> Dict[str, np.array]:
    """
    Return the templates of models (in a dict) to tell the structure
    The models need not to be trained
    :return: Dict[str,np.array]
    """
    net = Net()
    return Converter.cvtr().convert_nn_to_dict_nparray(net)


def training(models: Dict[str, np.array], init_flag: bool = False, DataStorage=None, SystemMeasurement=None,
             similarity_score_treshold=17, order=True, overall_score_threshold=0) -> Dict[str, np.array]:
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

    dm = DataManger(int(TrainingMetaData.num_training_data / 4), last_accuracy)
    print("DataStorage label accuracy:", last_accuracy)
    data_object_for_training = dm  # instance of DataManager object

    # returns trained neural network,

    bhattacharya_distance = (dm.dataset_bhattacharya_distance)
    similarity_score = bhattacharya_distance * 1000
    DataStorage.simialrity_scores.append(similarity_score)

    print("Similarity Score: ", similarity_score)
    print("Bhattacharya Distance: ", bhattacharya_distance)

    system_score, system_overide = SystemMeasurement.getDefaultSystemScore(DataStorage)
    DataStorage.system_scores.append(system_score)

    if (system_overide):
        print("SYSTEM PERFORMANCE OVERIDE")
    else:
        print("PREV ROUND TIME:", DataStorage.round_time[-1])
        print("SYSTEM SCORE:", system_score)

    if not system_overide:
        overall_score = similarity_score * 1000000 / system_score
        DataStorage.overall_scores.append(overall_score)
        if overall_score < overall_score_threshold:
            return models, None, None, None
    else:
        DataStorage.overall_scores.append(0)

    trained_net, round_loss, train_dataset_tensors, trainset_dataset_PIL = execute_ic_training(data_object_for_training,
                                                                                               net, criterion,
                                                                                               optimizer)

    models = Converter.cvtr().convert_nn_to_dict_nparray(trained_net)
    return models, round_loss, train_dataset_tensors, trainset_dataset_PIL


def compute_performance(models: Dict[str, np.array], testdata, is_local: bool) -> float:
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

    labels_predicted_correctly = [0] * 10
    labels_total = [0] * 10

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
        labels_accuracy.append(labels_predicted_correctly[i] / labels_total[i])

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


def write_analysis(DataStorage, SystemMeasurement, name):
    print("\n\WRITING ANALYSIS")

    folder_name = './test_files_' + name + '/'

    try:
        os.mkdir(folder_name)
    except:
        pass

    print("Folder Name", folder_name)

    # make df for local accuracies and global accuracies
    local_accuracies = DataStorage.get_local_accuracies()
    df_local_accuracies = pd.DataFrame({'Local Accuracies': local_accuracies, 'Round time': DataStorage.round_time})
    df_local_accuracies.to_csv(folder_name + '/model_local_accuracy_' + name + '.csv')

    df_round_participation_list = pd.DataFrame({'Round participation': DataStorage.participation_list})
    df_round_participation_list.to_csv(folder_name + '/round_participation_list_' + name + '.csv')

    df_global_accuracies = pd.DataFrame({'Global Accuracies': DataStorage.get_global_accuracies()})
    df_global_accuracies.to_csv(folder_name + '/model_global_accuracy_' + name + '.csv')

    df_scores = pd.DataFrame(
        {"Simialirity Scores": DataStorage.simialrity_scores, "System Scores": DataStorage.system_scores,
         "Overall Scores:": DataStorage.overall_scores})
    df_scores.to_csv(folder_name + '/model_scores_' + name + '.csv')

    df_skip_round_times = pd.DataFrame({'Skip Round Time': DataStorage.skip_round_time})
    df_skip_round_times.to_csv(folder_name + '/skip_round_time_' + name + '.csv')

    df_cpu_ram_utilisation = pd.DataFrame({'Average CPU Utilisation': SystemMeasurement.cpu_average_utilisation,
                                           "Average RAM Utilisation": SystemMeasurement.ram_average_utilisation})
    df_cpu_ram_utilisation.to_csv(folder_name + '/system_av_utilisation_' + name + '.csv')

    labels_accuracy = DataStorage.label_accuracy

    df = pd.DataFrame(labels_accuracy)
    df.to_csv(folder_name + "local_label_accuracy_" + name + ".csv")

    df_global_label_accuracies = pd.DataFrame(DataStorage.global_label_accuracy)
    df_global_label_accuracies.to_csv(folder_name + "/global_label_accuracy.csv")

    file_process_time = open(folder_name + "/file_process_time_" + name + ".txt", "w")
    file_process_time.write("Process start time: " + str(DataStorage.process_start_time) + "\n")
    file_process_time.write("Process end time:  " + str(DataStorage.process_end_time) + "\n")
    file_process_time.write("Full process time: " + str(DataStorage.process_whole_time))
    file_process_time.close()

    print("DONE\n")


def run_process(name, rounds_arg, overall_score_arg):
    import fl_main.agent.communication_client as communication_client

    print("TRAINING COUNT:", rounds_arg)
    print("OVERALL SCORE TRESHOLD:", overall_score_arg)

    logging.basicConfig(level=logging.INFO)
    logging.info('--- Heterogeneity Aware FL with client level intelligenece ---')

    fl_client = Client()
    logging.info(f'--- Your IP is {fl_client.agent_ip} ---')

    # for starting system measurement thread
    sm = SystemMeasurement(name)
    sys_thread = threading.Thread(target=sm.start_measurement)
    sys_thread.start()

    # Create a set of template models (to tell the shapes)
    initial_models = training(dict(), init_flag=True)

    # Sending initial models
    fl_client.send_initial_model(initial_models)

    # Starting FL client
    fl_client.start_fl_client()

    training_count = 0
    gm_arrival_count = 0
    skip_count = 0

    # number of rounds of training to run
    training_count_threshold = rounds_arg

    # similarity score treshold for client participation
    overall_score_threshold = overall_score_arg

    DataStorage = ds()
    AD = Analyser()

    communication_client.send("WELCOME MESSAGE!")

    while judge_termination(training_count, gm_arrival_count, training_count_threshold):

        print("\n\t TRAINING COUNT: ", training_count, "\n")
        sm.start_round()
        time_start = datetime.datetime.now()

        # Wait for Global models (base models)
        global_models = fl_client.wait_for_global_model()
        gm_arrival_count += 1

        # Global Model evaluation (id, accuracy)
        global_model_performance_data = compute_performance(global_models, prep_test_data(), False)
        # add model to add global model performance data to database

        DataStorage.add_global_label_accuracy(global_model_performance_data[1])

        DataStorage.add_global_accuracy(global_model_performance_data[0])  # this is lagged by one

        # Training
        models, round_loss, train_dataset_tensors, trainset_dataset_PIL = training(global_models,
                                                                                   DataStorage=DataStorage,
                                                                                   SystemMeasurement=sm,
                                                                                   overall_score_threshold=overall_score_threshold)

        training_count += 1

        if (round_loss == None):  # condition for checking if client is not participating
            # Sending initial models
            logging.info(f'--- FAILED SIMILARITY SCORE THRESHOLD, SKIPPING CURRENT ROUND ---')
            communication_client.send_non_participation_message()
            # fl_client.send_no_models()
            print("Message Global Models Size Of", sys.getsizeof(global_models))
            fl_client.send_initial_model(global_models, num_samples=0,
                                         perf_val=-1)  # sending sample if similarity score treshold failed
            skip_count += 1
            time_end = datetime.datetime.now()
            time_difference = (time_end - time_start).microseconds
            DataStorage.skip_round_time.append(time_difference)
            print("SKIP ROUND TIME TAKEN:", time_difference)
            DataStorage.participation_list.append(False)
            sm.end_round()
            continue

        logging.info(f'--- Training Done ---')

        # Local Model evaluation (id, accuracy)
        print("Message Global Models Size Of", sys.getsizeof(global_models))
        accuracy, labels_accuracy = compute_performance(models, prep_test_data(), True)
        DataStorage.add_label_accuracy(labels_accuracy)
        DataStorage.add_local_accuracy(accuracy)
        fl_client.send_trained_model(models, int(TrainingMetaData.num_training_data), accuracy)
        DataStorage.participation_list.append(True)
        time_end = datetime.datetime.now()
        time_difference = (time_end - time_start).microseconds
        DataStorage.round_time.append(time_difference)
        print("TRAINING ROUND TIME TAKEN:", time_difference)
        sm.end_round()

    DataStorage.end_full()
    write_analysis(DataStorage, sm, name)

    folder_name = './test_files_' + name + '/'

    sm.write_analysis(folder_name)  # system manager anaylysis
    communication_client.send_deregister_message()
    print("SENT DEREGISTER MESSAGE")
    sys.exit()


if __name__ == '__main__':
    try:
        name = sys.argv[3]
        rounds_arg = int(sys.argv[4])
        overall_score_arg = int(sys.argv[5])
    except:
        name = "NoName"
        rounds_arg = 25
        overall_score_arg = 0

    run_process(name, rounds_arg, overall_score_arg)


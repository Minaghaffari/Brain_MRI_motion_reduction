import os
import copy
import numpy as np
from random import shuffle
import tables

from configuration import config



def add_data(x_list, y_list, data_file, index):
    
    data, truth = data_file.root.data[index], data_file.root.truth[index, 0]     
    truth = truth[np.newaxis]

    x_list.append(data)
    y_list.append(truth)


def convert_data(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    print (x.shape , "    ", y.shape)      
    return x, y



def data_generator(data_file, batch_size=1, n_labels=1, labels=None):

    nb_samples = data_file.root.data.shape[0]
    sample_list = list(range(nb_samples))
    shuffle(sample_list)

    while True:
        x_list = list()
        y_list = list()
        
        index_list = copy.copy(sample_list)


        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index)
            if len(x_list) == batch_size: # or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list, n_labels=n_labels, labels=labels)
                x_list = list()
                y_list = list()


def get_number_of_steps(data_file, batch_size):
    n_samples = data_file.root.data.shape[0]
    if np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1



def get_training_and_validation_generators(training_data_file, training_batch_size, n_labels, 
                                           labels, validation_data_file, validation_batch_size):

    training_generator   = data_generator(training_data_file, training_batch_size, n_labels, labels)
    validation_generator = data_generator(validation_data_file, validation_batch_size, n_labels, labels)
    num_training_steps = get_number_of_steps(training_data_file, training_batch_size)
    num_validation_steps = get_number_of_steps(validation_data_file, validation_batch_size)

    return training_generator, validation_generator, num_training_steps, num_validation_steps



if __name__ == "__main__":

    training_data_file_opened = tables.open_file(config["training_data_file"]  , readwrite="r")
    validation_data_file_opened = tables.open_file(config["validation_data_file"]  , readwrite="r")


    training_generator, validation_generator, num_training_steps, num_validation_steps = get_training_and_validation_generators(
            training_data_file_opened, config["trainig_batch_size"], config["n_labels"], 
            config["labels"], validation_data_file_opened, config["validation_batch_size"])


    print (num_training_steps)
    print (num_validation_steps)

    for x,y in training_generator:
     print (x.shape)
     print (y.shape)
    

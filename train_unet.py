import os
import tables
import numpy as np
from unet import unet_model_3d
from training import train_model, model_load
from generator import get_training_and_validation_generators
from configuration import config


class train_network():
    def __init__(self):
        self.patch_shape = config["patch_shape"]

    def train(self, config):
        model = unet_model_3d(input_shape=config["input_shape"], n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"], n_base_filters=config["n_base_filters"])

        training_data_file_opened = tables.open_file(
            config["training_data_file"], readwrite="r")
        validation_data_file_opened = tables.open_file(
            config["validation_data_file"], readwrite="r")

        train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
            training_data_file_opened, config["trainig_batch_size"], config["n_labels"],
            config["labels"], validation_data_file_opened, config["validation_batch_size"])

        train_model(model=model,
                    model_file=config["model_file"],
                    training_generator=train_generator,
                    validation_generator=validation_generator,
                    steps_per_epoch=n_train_steps,
                    validation_steps=n_validation_steps,
                    initial_learning_rate=config["initial_learning_rate"],
                    learning_rate_drop=config["learning_rate_drop"],
                    learning_rate_patience=config["patience"],
                    early_stopping_patience=config["early_stop"],
                    n_epochs=config["n_epochs"])


if __name__ == "__main__":
    net_train = train_network()
    net_train.train(config)

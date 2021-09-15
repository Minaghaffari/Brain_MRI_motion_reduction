import os
import glob
import pickle
import numpy as np
from random import shuffle
from configuration import config

class write_img_paths():
    def __init__(self):
        self.n_test = 2        
        self.n_training = 5

    def write_img_paths(self):
        sample_list = []
        for files in glob.glob(os.path.join(config ["Motion_simulated_dir"], "*")):
            sample_list.append(files)
        shuffle(sample_list)

        test_list = sample_list[0:self.n_test]
        training_list = []
        for i in range(config["n_fold_crossValidation"]):
            training_list.append(
                sample_list[self.n_test + self.n_training * i: self.n_test + self.n_training * (i+1)])
            print("training_list=======>", len(training_list[i]))

        with open(config["test_files"], "wb") as fp:
            pickle.dump(test_list, fp)

        with open(config["training_files"], "wb") as fp:
            pickle.dump(training_list, fp)


if __name__ == "__main__":
    paths_write = write_img_paths()
    paths_write.write_img_paths()



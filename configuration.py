import os


config = dict ()

config["image_shape"] = (260, 311, 260)         #for HCP images
config["patch_shape"] = (128,128,128)
config["training_data_file"] = os.path.abspath("training_data.h5")
config["validation_data_file"] = os.path.abspath("validation_data.h5")
config["overwrite"]= False
config["nb_channels"] = 1
config["labels"] = (1,)  


#model related
config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
config["n_labels"] = len(config["labels"])
config["n_base_filters"] = 16
config["initial_learning_rate"] = 5e-4



#data generator related 
config["trainig_batch_size"] = 1
config["validation_batch_size"] = 1



#train related
config["initial_learning_rate"] = 2e-4
config["learning_rate_drop"] = 0.5  
config["n_epochs"] = 300  
config["patience"] = 10  
config["early_stop"] = 50  



# files related
config["n_fold_crossValidation"] = 1
config["crossValidation_lot"] = 0
config["training_files"] = "training_dirs.txt"
config["test_files"] = "test_dirs.txt"
config["model_file"] = os.path.abspath("3Dunet")



config ["ground_truth_dir"] =  "./Data_motionFree"
config ["Motion_simulated_dir"] = "./Data_motioned"


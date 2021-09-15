import os
import tables
import numpy as np

from keras.layers import Input
from keras.engine import Model
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
#from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format("channels_first")

from data_loader import DataLoader
from configuration import config
from discriminator import discriminator
#from Unet_plus import Unet_plus
from unet import unet_model_3d

class vox_to_vox():
    def __init__(self):
        self.img_rows = config["patch_shape"][0]
        self.img_cols = config["patch_shape"][1]
        self.img_slices = config["patch_shape"][2]
        self.img_shape = config["patch_shape"]

        # Configure data loader
        self.data_loader = DataLoader(dataset_path=config["training_data_file"], batch_size=config["trainig_batch_size"] )


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (1, patch, patch, patch)

        # Number of filters in the first layer of G and D and optimizer definition
        self.gf = 16
        self.df = 64
        optimizer = Adam(learning_rate=config["initial_learning_rate"], beta_1=0.5)


        # Input images and their conditioning images
        img_A = Input(shape=config["input_shape"])    #img_A is the ground_truth 
        img_B = Input(shape=config["input_shape"])    #img_B is the noisy input



        # Build and compile the discriminator
        self.discriminator = discriminator(img_A, img_B, n_filters=self.df )
        # self.discriminator = self.model_load ("./Models/D_model_imgNorm99.h5")
        self.discriminator.compile(loss='mse', optimizer=optimizer)

        # Build the generator
#        self.generator = unet_model_3d()  #Unet_plus() 
        self.generator = self.model_load ("./unet.h5")
        self.generator.name = "gen_model"

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[100, 100], optimizer=optimizer)




    def model_load (self, model_file):
        print("Loading pre-trained model")
        from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        custom_objects = {"InstanceNormalization":InstanceNormalization}
        return load_model(model_file, custom_objects=custom_objects)



    def train(self, epochs, batch_size=1):

        # Adversarial loss ground truths
#        valid = np.ones((batch_size,) + self.disc_patch)
#        fake = np.zeros((batch_size,) + self.disc_patch)

        fake   = np.random.uniform(low=0, high=0.3, size =((batch_size,) + self.disc_patch) )
        valid  = np.random.uniform(low=0.7, high=1.2, size = ((batch_size,) + self.disc_patch))

        epochs_d_loss = []
        epochs_g_loss = []
        n_batches = 1

        for epoch in range(epochs):
            current_epoch_d_loss = 0
            current_epoch_g_loss = 0
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                print ("DIS trainable befor training DIS" , self.discriminator.trainable)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------
                print ("DIS trainable befor training combined" , self.discriminator.trainable)
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                current_epoch_d_loss += d_loss
                current_epoch_g_loss += g_loss[0]
                n_batches = batch_i


            epochs_d_loss.append(current_epoch_d_loss / n_batches)
            epochs_g_loss.append(current_epoch_g_loss / n_batches)

            #write to a log file
            with open (config["loss_logfile"] , "a") as fp:
                loss_log = "%d        %f        %f\n" %(epoch, epochs_d_loss[-1], epochs_g_loss[-1] )
                fp.write(loss_log)
        
            #save model
            G_model_path = "./Models_unet_cGAN_tl_100_100/G_model%d.h5" %(epoch)
            D_model_path = "./Models_unet_cGAN_tl_100_100/D_model%d.h5" %(epoch)
            self.generator.save(G_model_path)
            self.discriminator.save(D_model_path)


if __name__ == '__main__':
    gan = vox_to_vox()
    gan.train(epochs=100, batch_size=1)            




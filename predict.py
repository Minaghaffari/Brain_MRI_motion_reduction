import os
import glob
import pickle
import numpy as np
import nibabel as nib
from keras import backend as K
from keras.models import load_model
from configuration import config
from nilearn.image import new_img_like



class predict_patch():
    def __init__(self):
        self.img_rows = config["patch_shape"][0]
        self.img_cols = config["patch_shape"][1]
        self.img_slices = config["patch_shape"][2]
        self.patch_shape = config["patch_shape"]
        self.img_shape = config["image_shape"]

    def crop_data (self, image_data):
        passes_threshold = (image_data > 0)
        coords = np.array(np.where(passes_threshold))
        start = coords.min(axis=1)
        end = coords.max(axis=1) + 1
        slices = [slice(s, e) for s, e in zip(start, end)]
        return slices    

    def model_load (self, model_file):
        from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        custom_objects = {"InstanceNormalization":InstanceNormalization}     
        return load_model(model_file, custom_objects=custom_objects)   


    def patch_gen (self, img_ary, img_ary_scaled):           
        img_slices = self.crop_data (img_ary)
        patch_slices_start = self.patch_slices_start_gen (img_slices)
        pathes_slices = self.patch_slices_gen (patch_slices_start)
        image_patchs_data = []
        for patch_slice in pathes_slices:            
            img_patch = img_ary_scaled[tuple(patch_slice)]
            image_patchs_data.append(img_patch)
        return pathes_slices, image_patchs_data
        

    def patch_slices_start_gen (self, slices):
        slices_size = [s.stop -s.start for s in slices]
        patch_slices_start = []
        for i in range (len(slices_size)):   
            patch_slices_start.append([]) 
            mid = int((slices[i].start + slices[i].stop)/2)
            if slices_size [i] <= (self.patch_shape[i]*2):
                patch_slices_start[i].append (slices[i].start)
                patch_slices_start[i].append (slices[i].stop - self.patch_shape[i] )
            else:
                patch_slices_start[i].append(slices[i].start )
                patch_slices_start[i].append(int(mid - (self.patch_shape[i]) /2 ) )
                patch_slices_start[i].append( slices[i].stop - self.patch_shape[i])
        return  patch_slices_start       


    def patch_slices_gen (self, slices_starts):
        patches_slice_starts = []
        for i in range (len(slices_starts[0])):
            for j in range (len(slices_starts[1])):
                for k in range (len(slices_starts[2])):
                    patches_slice_starts.append([slices_starts[0][i], slices_starts[1][j], slices_starts[2][k]])
        patches_slices = []
        for patches_start in patches_slice_starts:
            patches_slices.append ([slice(s, s+self.patch_shape [idx]) for idx, s in enumerate(patches_start)])     
        return patches_slices 



    def patchs_predict (self, model, data_patch_list):
        prediction_list = []
        for data_patch in data_patch_list:
            data_patch = np.expand_dims(data_patch , axis =0)  
            data_patch = np.expand_dims(data_patch , axis =0)
            pred = model.predict (data_patch)
            prediction_list.append (pred[0,0,:,:,:])
        return prediction_list


   
    def  attach_patches (self, prediction_list, patches_slices, img_ary):
        prediction = np.zeros (self.img_shape)-1
        img_slices = self.crop_data (img_ary)
        prediction[tuple(img_slices)] = 0        
        overlaps = np.zeros (self.img_shape, dtype=np.uint8)
        for patch_slice, patch_prediction  in zip (patches_slices, prediction_list):
            prediction[tuple(patch_slice)] += patch_prediction
            overlaps[tuple(patch_slice)] += 1
        overlaps [overlaps == 0] = 1
        final_prediction = prediction / overlaps
        return final_prediction
  



if __name__ == "__main__":
    predict_patch_MRI = predict_patch()
    model = predict_patch_MRI.model_load(model_file = "./unet_std.h5")
    with open("./test_dirs.txt", "rb") as fp:
        all_test_files_list = pickle.load(fp)
    for img_path in all_test_files_list:
        subject_id = (img_path.split("/")[-1])   
        pred_path = "./predictions/%s" %subject_id
        if not os.path.exists (pred_path):
            img = nib.load (img_path) 
            img_ary= img.get_fdata()
            img_std = np.std(img_ary)
            img_ary_scaled = img_ary/(img_std + 0.000001)
            pathes_slices, image_patchs_data = predict_patch_MRI.patch_gen (img_ary, img_ary_scaled)
            paths_prediction_list = predict_patch_MRI.patchs_predict(model, image_patchs_data )
            predicted_img_ary  = predict_patch_MRI.attach_patches (paths_prediction_list, pathes_slices, img_ary)
            predicted_img_ary_rescale = predicted_img_ary * img_std
            pred_img = new_img_like (img,predicted_img_ary_rescale)
            nib.save (pred_img, pred_path)











import os
import glob
import tables
import pickle
import numpy as np
import nibabel as nib
from configuration import config


class write_data_h5():
    def __init__(self):
        self.patch_shape = config["patch_shape"]

    def crop_data(self, image_data):

        # returns the slices for cropping an image_data as much as possible
        passes_threshold = (image_data > 0)
        coords = np.array(np.where(passes_threshold))
        start = coords.min(axis=1)
        end = coords.max(axis=1) + 1
        slices = [slice(s, e) for s, e in zip(start, end)]
        return slices

    def training_validation_split(self, file_path, crossValidation_lot):
        with open(file_path, "rb") as fp:
            paths_list = pickle.load(fp)
        paths = paths_list[crossValidation_lot]
        n_validation = 1
        validation = paths[0:n_validation]
        training = paths[n_validation:]
        return training, validation

    def fetch_training_data_files(self, subject_paths):
        training_data_files = list()
        for subject_path in subject_paths:
            print(subject_path)
            subject_id = os.path.basename(subject_path)
            print(subject_id)
            subject_files = []
            subject_files.append(subject_path)
            ground_truth_path = config ["ground_truth_dir"] + "/%s.nii.gz" % subject_id.split("_")[0]
            print(ground_truth_path)
            subject_files.append(ground_truth_path)
            training_data_files.append(tuple(subject_files))
        return training_data_files

    def create_data_file(self, out_file, n_channels, n_samples, image_shape):
        hdf5_file = tables.open_file(out_file, mode='w')
        filters = tables.Filters(complevel=5, complib='blosc')
        data_shape = tuple([0, n_channels] + list(image_shape))
        truth_shape = tuple([0, 1] + list(image_shape))
        data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                               filters=filters, expectedrows=n_samples)
        truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.Float32Atom(), shape=truth_shape,
                                                filters=filters, expectedrows=n_samples)
        return hdf5_file, data_storage, truth_storage

    def read_image_files(self, image_files):
        image_data_list = list()
        scaled_image_data_list = list()
        MS_img = nib.load(os.path.abspath(image_files[0]))
        MS_img_ary = MS_img.get_fdata()
        MS_img_ary_std = np.std(MS_img_ary)

        for image_file in image_files:
            print("Reading: {0}".format(image_file))
            image = nib.load(os.path.abspath(image_file))
            img_data = image.get_fdata()
            image_data_list.append(img_data)
            scaled_img_data = img_data/(MS_img_ary_std+0.000001)
            scaled_image_data_list.append(scaled_img_data)
        return image_data_list, scaled_image_data_list

    def add_data_to_storage(self, data_storage, truth_storage, subject_data, n_channels):
        data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
        truth_storage.append(np.asarray(subject_data[n_channels])[
                             np.newaxis][np.newaxis])

    def patch_slices_start_gen(self, slices, patch_shape, image_shape):
        slices_size = [s.stop - s.start for s in slices]
        patch_slices_start = []

        for i in range(len(slices_size)):
            patch_slices_start.append([])
            if slices_size[i] <= (patch_shape[i]*2):
                patch_slices_start[i].append(slices[i].start)
                patch_slices_start[i].append(slices[i].stop - patch_shape[i])
            else:
                mid = int((slices[i].start + slices[i].stop)/2)
                patch_slices_start[i].append(slices[i].start)
                patch_slices_start[i].append(int(mid - (patch_shape[i]) / 2))
                patch_slices_start[i].append(slices[i].stop - patch_shape[i])
        return patch_slices_start

    def patch_slices_gen(self, slices_starts, patch_shape):
        patches_slice_starts = []
        for i in range(len(slices_starts[0])):
            for j in range(len(slices_starts[1])):
                for k in range(len(slices_starts[2])):
                    patches_slice_starts.append(
                        [slices_starts[0][i], slices_starts[1][j], slices_starts[2][k]])

        patches_slices = []
        for patches_start in patches_slice_starts:
            patches_slices.append([slice(s, s + patch_shape[idx])
                                   for idx, s in enumerate(patches_start)])
        return patches_slices

    def write_image_data_to_file(self, image_files, data_storage, truth_storage, patch_shape, image_shape, n_channels):
        for set_of_files in image_files:
            images_data, scaled_image_data_list = self.read_image_files(
                set_of_files)
            # calculate the slice  for motion corrupted images
            img_slices = self.crop_data(images_data[0])
            patch_slices_starts = self.patch_slices_start_gen(
                img_slices, patch_shape, image_shape)
            pathes_slices = self.patch_slices_gen(
                patch_slices_starts, patch_shape)

            for patch_slice in pathes_slices:
                images_patchs_data = []
                motioned_data_patch = scaled_image_data_list[0]
                img_patch_ary = motioned_data_patch[tuple(patch_slice)]
                images_patchs_data.append(img_patch_ary)
                ground_truth_patch = scaled_image_data_list[1]
                gt_patch_ary = ground_truth_patch[tuple(patch_slice)]
                images_patchs_data.append(gt_patch_ary)
                self.add_data_to_storage(
                    data_storage, truth_storage, images_patchs_data, n_channels)
        return data_storage, truth_storage

    def write_data_to_file(self, training_data_files, out_file, patch_shape, image_shape):
        # x12 for patches being extracted
        n_samples = 12 * len(training_data_files)
        n_channels = len(training_data_files[0]) - 1
        # number of modalities (here we have only T1-w MRI so n_channel=1)
        print("n_channels ====>", n_channels)

        hdf5_file, data_storage, truth_storage = self.create_data_file(
            out_file, n_channels, n_samples, patch_shape)
        self.write_image_data_to_file(
            training_data_files, data_storage, truth_storage, patch_shape, image_shape, n_channels)

        hdf5_file.close()
        return out_file


if __name__ == "__main__":
    if config["overwrite"] or not os.path.exists(config["training_data_file"]):

        write_data_patches = write_data_h5()
        training, validation = write_data_patches.training_validation_split(
            config["training_files"], config["crossValidation_lot"])

        training_files = write_data_patches.fetch_training_data_files(training)
        write_data_patches.write_data_to_file(
            training_files, config["training_data_file"], config["patch_shape"], config["image_shape"])

        validation_files = write_data_patches.fetch_training_data_files(
            validation)
        write_data_patches.write_data_to_file(
            validation_files, config["validation_data_file"], config["patch_shape"], config["image_shape"])

import os
import glob
import numpy as np
import copy as cp
import pandas as pd
import nibabel as nib
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def ssim(gt, pred):
    return structural_similarity(gt,pred, multichannel=True, data_range=gt.max())

def nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def psnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())





def evaluate_images (main_path, preds_path):
    rows_nmse = []
    rows_psnr = []
    rows_ssim = []
    subject_ids = []

    for files in glob.glob (os.path.join(preds_path, "*")):
        file_name = os.path.basename(files)
        img_ID = (os.path.basename(files)).split(".")[0]
        print(img_ID)
        MS_path = os.path.join(main_path, "dataset/HCP_noBET_cropped_motioned_bet/%s.nii.gz" %img_ID )
        GT_path = os.path.join(main_path, "dataset/HCP_original/%s.nii.gz" %img_ID)

        subject_ids.append(img_ID)        

        pred_img = nib.load(files)
        MS_img = nib.load(MS_path)
        GT_img = nib.load(GT_path)
        

        pred_ary = pred_img.get_fdata()
        MS_ary = MS_img.get_fdata()
        GT_ary = GT_img.get_fdata()


        mask = np.zeros(GT_ary.shape)
        mask [GT_ary != 0] = 1


        pred_ary [mask == 0] = 0
        MS_ary [mask == 0] = 0



        NMSEs = [ nmse(GT_ary, MS_ary), nmse(GT_ary, pred_ary)]
        PSNRs = [psnr(GT_ary, MS_ary), psnr(GT_ary, pred_ary)]
        SSIMs = [ssim(GT_ary, MS_ary), ssim (GT_ary, pred_ary)]

        rows_nmse.append(NMSEs)
        rows_psnr.append(PSNRs)
        rows_ssim.append(SSIMs)

    return subject_ids, rows_nmse, rows_psnr, rows_ssim


def writeCSV (subject_ids, rows_nmse, rows_psnr, rows_ssim):
    header_nmse = ["NMSE MS vs GT", "NMSE Pred vs GT"]
    header_psnr = ["PSNR MS vs GT", "PSNR Pred vs GT"]
    header_ssim = ["SSIM MS vs GT", "SSIM Pred vs GT"]

    mean_nmse = np.mean(np.array (rows_nmse), axis= 0)
    rows_nmse.append (mean_nmse)

    mean_psnr = np.mean(np.array (rows_psnr), axis= 0)
    rows_psnr.append (mean_psnr)

    mean_ssim = np.mean(np.array (rows_ssim), axis= 0)
    rows_ssim.append (mean_ssim)

    subject_ids.append ("Average Score")

    DataFrame_nmse = pd.DataFrame.from_records(rows_nmse, columns=header_nmse, index=subject_ids)
    DataFrame_psnr= pd.DataFrame.from_records(rows_psnr, columns=header_psnr, index=subject_ids)
    DataFrame_ssim = pd.DataFrame.from_records(rows_ssim, columns=header_ssim, index=subject_ids)


    DataFrame_nmse_filePath = "./NMSEs.csv"
    DataFrame_psnr_filePath = "./PSNRs.csv"
    DataFrame_ssim_filePath = "./SSIMs.csv"

    DataFrame_nmse.to_csv(DataFrame_nmse_filePath)
    DataFrame_psnr.to_csv(DataFrame_psnr_filePath)
    DataFrame_ssim.to_csv(DataFrame_ssim_filePath)

    return mean_nmse






if __name__ == "__main__":
    main_path = os.path.abspath(os.path.join(__file__ ,"../../../.."))
    subject_ids, rows_nmse, rows_psnr, rows_ssim = evaluate_images (main_path, "./predictions" )
    mean_nmse = writeCSV (subject_ids, rows_nmse, rows_psnr, rows_ssim)




import os
import argparse
import numpy as np
import pandas as pd
import pickle

import SimpleITK as sitk

from PIL import Image

import matplotlib
matplotlib.use('TkAgg') # not show is Agg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def show_one():
    files = ['list_of_seg_per_patient.pickle', 'raw_pred_boxes_list.pickle']

    root = '/home/cg/medicaldetectiontoolkit/experiments/tooth/fold_0'
    info_df_path = '/home/cg/medicaldetectiontoolkit/tooth_dataset/preprocessing/info_df.pickle'

    pred_segs_pickle_path = os.path.join(root, files[0])    	
    with open(pred_segs_pickle_path, 'rb') as f:
        pred_segs_pickle = pickle.load(f)
    
    boxes_pickle_path = os.path.join(root, files[1])    	
    with open(boxes_pickle_path, 'rb') as f:
        boxes_pickle = pickle.load(f)

    with open(info_df_path, 'rb') as f:
        info_df_pickle = pickle.load(f)

    raw_pids = info_df_pickle['raw_pid']
    pid = pred_segs_pickle[0][1]
    raw_pid = raw_pids[pid]
    
    print('pid = {}, raw_pid = {}'.format(pid, raw_pid))
    ori_data_path = '/home/cg/medicaldetectiontoolkit/tooth_dataset/preprocessing/2_img.npy'
    ori_seg_path = '/home/cg/medicaldetectiontoolkit/tooth_dataset/preprocessing/2_msk.npy'
    ori_data = np.transpose(np.load(ori_data_path), axes=(1, 2, 0))
    ori_seg = np.transpose(np.load(ori_seg_path), axes=(1, 2, 0))
    print('ori data =', ori_data.shape)
    print('ori seg =', ori_seg.shape)

    # print()
    print('pred_segs =', pred_segs_pickle[0][0].shape)
    pred_segs = pred_segs_pickle[0][0].squeeze().astype(np.float32) # (4, 400, 400, 272)
    pred_segs = pred_segs[0, :, :, :] # (400, 400, 272)
    print(pred_segs.shape)
    print(np.unique(pred_segs))
    
    # assert -1>0

    slice_num = 100
    img = ori_data[:, :, slice_num]
    lbl = ori_seg[:, :, slice_num]
    pred = pred_segs[:, :, slice_num]

    matplotlib.rcParams.update({'font.size': 3})
    plt.rcParams['figure.dpi'] = 300

    plt.subplot(2,4,1)
    plt.title("image")
    plt.imshow(img, cmap='gray')

    plt.subplot(2,4,2)
    plt.title("mask")
    plt.imshow(lbl, cmap='gray')

    plt.subplot(2,4,3)
    plt.title("image & gt mask")
    lbl_masked = np.ma.masked_where(lbl == 0, lbl)
    plt.imshow(img, cmap='gray')
    plt.imshow(lbl_masked, alpha=0.5)

    plt.subplot(2,4,4)
    plt.title("pred mask")
    plt.imshow(pred, cmap='gray') # tab20c

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=-0.3)
    # plt.tight_layout()
    plt.show()

    # save seg to nii.gz
    out = sitk.GetImageFromArray(pred_segs)
    file_name = 'pred_{}.nii.gz'.format(raw_pid)
    sitk.WriteImage(out, file_name)
    print('! save {}'.format(file_name))

def load_pickle():
    files = ['list_of_seg_per_patient_0.pickle', 'raw_pred_boxes_list_0.pickle']
    files01 = ['list_of_seg_per_patient_1.pickle', 'raw_pred_boxes_list_1.pickle']
    files02 = ['list_of_seg_per_patient_2.pickle', 'raw_pred_boxes_list_2.pickle']
    files03 = ['list_of_seg_per_patient_3.pickle', 'raw_pred_boxes_list_3.pickle']

    root = '/home/cg/medicaldetectiontoolkit/experiments/tooth/fold_0'
    info_df_path = '/home/cg/medicaldetectiontoolkit/tooth_dataset/preprocessing/info_df.pickle'

    segs_pickle_path = os.path.join(root, files[0])    	
    with open(segs_pickle_path, 'rb') as f:
        segs_pickle = pickle.load(f)
    
    boxes_pickle_path = os.path.join(root, files[1])    	
    with open(boxes_pickle_path, 'rb') as f:
        boxes_pickle = pickle.load(f)

    segs_pickle_path01 = os.path.join(root, files01[0])    	
    with open(segs_pickle_path01, 'rb') as f:
        segs_pickle01 = pickle.load(f)
    
    boxes_pickle_path01 = os.path.join(root, files01[1])    	
    with open(boxes_pickle_path01, 'rb') as f:
        boxes_pickle01 = pickle.load(f)

    segs_pickle_path02 = os.path.join(root, files02[0])    	
    with open(segs_pickle_path02, 'rb') as f:
        segs_pickle02 = pickle.load(f)
    
    boxes_pickle_path02 = os.path.join(root, files02[1])    	
    with open(boxes_pickle_path02, 'rb') as f:
        boxes_pickle02 = pickle.load(f)

    segs_pickle_path03 = os.path.join(root, files03[0])    	
    with open(segs_pickle_path03, 'rb') as f:
        segs_pickle03 = pickle.load(f)
    
    boxes_pickle_path03 = os.path.join(root, files03[1])    	
    with open(boxes_pickle_path03, 'rb') as f:
        boxes_pickle03 = pickle.load(f)

    with open(info_df_path, 'rb') as f:
        info_df_pickle = pickle.load(f)

    pid = 2
    # print(len(boxes_pickle))
    # assert -1>0
    # pid = boxes_pickle[1][1]
    print('pid =', pid, '1001152328_20180910')
    
    # dfmask = info_df_pickle['pid'] == str(pid)
    # print('file name =', info_df_pickle[dfmask].iat[0,0])

    # pid_ori_data_path = info_df_pickle[dfmask].iat[0, 0]
    pid_ori_data_path = '/home/cg/medicaldetectiontoolkit/tooth_dataset/preprocessing/2_img.npy'
    pid_ori_seg_path = '/home/cg/medicaldetectiontoolkit/tooth_dataset/preprocessing/2_msk.npy'
    pid_ori_data = np.load(pid_ori_data_path)
    pid_ori_seg = np.load(pid_ori_seg_path) 
    print('ori data =', pid_ori_data.shape)
    print('ori seg =', pid_ori_seg.shape)

    # pred seg
    print(segs_pickle[0][0].shape)
    pid_segs = segs_pickle[0][0].squeeze()
    pid_segs01 = segs_pickle01[0][0].squeeze()
    pid_segs02 = segs_pickle02[0][0].squeeze()
    pid_segs03 = segs_pickle03[0][0].squeeze()

    pid_segs = np.concatenate((pid_segs, pid_segs01), axis=3)
    pid_segs02_03 = np.concatenate((pid_segs02, pid_segs03), axis=3)

    pid_segs = np.concatenate((pid_segs, pid_segs02_03), axis=2)
    print('pid_segs =', pid_segs.shape)

    pid_segs = np.transpose(pid_segs, axes=(0, 1, 4, 2, 3))

    print('pid_segs.shape =', pid_segs.shape)


    # iou
    max_iou = 0.85
    # max_iou_idx = []
    # gt = pid_ori_data[1, :, :]

    # for i in range(2):
    #     for j in range(4):
    #         iou_val = iou(pid_segs[i, j, :, :], gt, 1)
    #         if iou_val > max_iou:
    #             max_iou = iou_val
    #             max_iou_idx = [i, j]
    
    # print('max_iou = ', max_iou)
    # print('max_iou = ', max_iou_idx)
    
    # for i in range(280):
        # if(len(np.unique(pid_segs[0, 0, i, :, :])) > 1):
            # print(i)
            # print(np.unique(pid_segs[0, 0, i, :, :]))
    # pred_lbl = pid_segs[max_iou_idx[0], max_iou_idx[1], :, :]

    # show image
    slice_num = 185
    img = pid_ori_data[slice_num, :, :]
    lbl = pid_ori_seg[slice_num, :, :]

    pred_lbl = pid_segs[4] # latest checkpoint
    print('pred_lbl.shape =', pred_lbl.shape)
    save_pred_lbl = np.zeros(pred_lbl[0].shape)
    print('new_pred_lbl.shape =', save_pred_lbl.shape)

    for i in range(4):
        tmp = pred_lbl[i]
        save_pred_lbl[tmp>=1] = i+1
    print(np.unique(save_pred_lbl))
    
    out = sitk.GetImageFromArray(save_pred_lbl)
    sitk.WriteImage(out, 'out.nii.gz')
    print('! save out.nii.gz')

    # print('pred_lbl.shape =', pred_lbl.shape)
    # print('pred_unique_id = ', np.unique(pred_lbl))
    # print('gt_unique_id = ',np.unique(lbl))


    matplotlib.rcParams.update({'font.size': 3})
    plt.rcParams['figure.dpi'] = 300

    plt.subplot(2,4,1)
    plt.title("image")
    plt.imshow(img, cmap='gray')
    
    plt.subplot(2,4,2)
    plt.title("gt mask")
    plt.imshow(lbl, cmap='gray')

    plt.subplot(2,4,3)
    plt.title("image & gt mask")
    lbl_masked = np.ma.masked_where(lbl == 0, lbl)
    plt.imshow(img, cmap='gray')
    plt.imshow(lbl_masked, alpha=0.5)

    plt.subplot(2,4,4)
    plt.title("pred mask")
    plt.imshow(save_pred_lbl[slice_num, :, :], cmap='gray') # tab20c



    # plt.subplot(2,4,6)
    # plt.title("image & pred mask")
    # lbl_masked = np.ma.masked_where(pred_lbl == 0, pred_lbl)
    # plt.imshow(img, cmap='gray')
    # plt.imshow(lbl_masked, alpha=0.5)
    
    # plt.subplot(2,4,7)
    # plt.title("gt mask & pred mask")
    # # set theory
    # result_image = np.zeros(lbl.shape)

    # union = np.logical_or(lbl == 1, pred_lbl == 1)
    # intersection = np.logical_and(lbl == 1, pred_lbl == 1)
    # lbl_part = lbl - intersection
    # pred_lbl_part = pred_lbl - intersection

    # cp_intersection = np.logical_and(result_image == 0, intersection == 1)
    # cp_lbl_part = np.logical_and(result_image == 0, lbl_part == 1)
    # cp_pred_lbl_part = np.logical_and(result_image == 0, pred_lbl_part == 1)

    # # set val
    # result_image[cp_lbl_part] = 1
    # result_image[cp_pred_lbl_part] = 2
    # result_image[cp_intersection] = 3

    # im = Image.fromarray(np.uint8(result_image))

    # # change val to color
    # im.putpalette([
    #     0, 0, 0, # black background
    #     255, 0, 0, # index 1 is red -> lbl
    #     0, 255, 0, # index 2 is green -> pred_lbl
    #     255, 255, 0, # index 3 is yellow -> intersection
    # ])

    # plt.imshow(im)

    # plt.subplot(2,4,8)  
    # plt.title("IOU = {}".format(round(max_iou, 2)))

    # # Crop
    # newImage = im.crop((32, 240, 110, 320))
    # plt.imshow(newImage)

    # # red_patch = mpatches.Patch(color='red', label='Ground Truth')
    # red_patch = Line2D([], [], color="red", label='Ground Truth', marker='o', markerfacecolor="red", linewidth=0, markersize=4)
    # g_patch = Line2D([], [], color="green", label='Pred Label', marker='o', markerfacecolor="green", linewidth=0, markersize=4)
    # y_patch = Line2D([], [], color="yellow", label='Intersection', marker='o', markerfacecolor="yellow", linewidth=0, markersize=4)
    # plt.legend(handles=[red_patch, g_patch, y_patch])

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=-0.3)
    # plt.tight_layout()
    plt.show()
    # save_name = 'pid_{}_iou_{}.png'.format(pid, round(max_iou, 2))
    # save_name = 'pid_{}.png'.format(pid)
    # plt.savefig(save_name)
    # print('saved', save_name)

def iou(input, target, classes=1):
    """  compute the value of iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        iou: float, the value of iou
    """
    intersection = np.logical_and(target == classes, input == classes)
    # print(intersection.any())
    union = np.logical_or(target == classes, input == classes)
    iou = np.sum(intersection) / np.sum(union)
    return iou


if __name__ == '__main__':
    show_one()

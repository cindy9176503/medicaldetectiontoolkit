#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from pathlib import PurePath
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
from scipy.ndimage.measurements import label as lb
from scipy.ndimage.measurements import center_of_mass as com
import nrrd

from skimage.segmentation import clear_border

from experiments.tooth import configs
cf = configs.configs()


def pp_patient(inputs):
    # ix is index of data_dict in data_dicts
    ix, data_dict = inputs

    pid = ix
    print('processing', pid, data_dict)

    # read img and msk
    img = sitk.ReadImage(data_dict['image']) # h, w, c (y, x, z)
    msk = sitk.ReadImage(data_dict['mask']) # h, w, c (y, x, z)

    # check space and origin, must have same
    assert img.GetSpacing() == msk.GetSpacing() and img.GetOrigin() == msk.GetOrigin()

    # get img and msk as np array
    img = sitk.GetArrayFromImage(img).astype(np.float32)
    msk = sitk.GetArrayFromImage(msk).astype(np.uint8)

    # get new mask id (get increasing id)
    tmp_lbl_uiq = list(np.unique(msk))  
    print('tmp_lbl_uiq = ', tmp_lbl_uiq)
  
    new_idx = np.arange(0, len(tmp_lbl_uiq)) 
    for idx, mask_idx in enumerate(tmp_lbl_uiq):
        msk[msk == mask_idx] = new_idx[idx]  

    # get label
    lbl = list(np.unique(msk.flatten()).astype(int))
    print('new_idx = ', lbl)

    # save img and msk
    np.save(os.path.join(cf.pp_dir, '{}_img.npy'.format(pid)), img)
    np.save(os.path.join(cf.pp_dir, '{}_msk.npy'.format(pid)), msk)

    return {
        'pid': pid,
        'raw_pid': PurePath(data_dict['image']).parts[-1].split('.')[0],
        'class_target': [1 for _ in range(len(lbl)-1)],  # (len(lbl)-1) remove background
    }


def get_data_dicts(data_dir):
    img_dir = os.path.join(data_dir, 'img')
    lbl_dir = os.path.join(data_dir, 'label')
    img_pths = sorted(os.listdir(img_dir))
    lbl_pths = sorted(os.listdir(lbl_dir))

    data_dicts = []
    for img_pth, lbl_pth in zip(img_pths, lbl_pths):
        data_dict = {
            'image': os.path.join(img_dir, img_pth),
            'mask': os.path.join(lbl_dir, lbl_pth),
        }
        data_dicts.append(data_dict)
    return data_dicts


if __name__ == "__main__":
    # only select some data for testing
    data_dicts = get_data_dicts(cf.raw_data_dir)[:]
    print(data_dicts)
    print('data dicts:', len(data_dicts))

    os.makedirs(cf.pp_dir, exist_ok=True)

    df = pd.DataFrame(columns=['pid', 'raw_pid', 'class_target'])

    for inp in enumerate(data_dicts):
        pp_patient(inp)
        df.loc[len(df)] = pp_patient(inp)

    df.to_pickle(os.path.join(cf.pp_dir, 'info_df.pickle'))
    df.to_csv(os.path.join(cf.pp_dir, 'info_df.csv'))
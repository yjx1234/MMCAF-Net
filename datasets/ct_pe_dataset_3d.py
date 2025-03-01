import cv2
import h5py
import numpy as np
import os
import pickle
import torch
import util
import random
import albumentations
from scipy.ndimage.interpolation import rotate
import sys
#w
from ct.ct_pe_constants import *
from .base_ct_dataset import BaseCTDataset

class CTPEDataset3d(BaseCTDataset):
    def __init__(self, args, phase, is_training_set = True):
        super(CTPEDataset3d, self).__init__(args.data_dir, args.img_format, is_training_set = is_training_set)
        self.phase = phase
        self.resize_shape = args.resize_shape
        self.is_test_mode = not args.is_training
        self.pe_types = args.pe_types
        #w
        self.crop_shape = args.crop_shape
        self.do_hflip = self.is_training_set and args.do_hflip
        self.do_vflip = self.is_training_set and args.do_vflip
        self.do_rotate = self.is_training_set and args.do_rotate
        self.do_jitter = self.is_training_set and args.do_jitter
        #w 默认为真
        self.pixel_dict = {
            'min_val':CONTRAST_HU_MIN,
            'max_val':CONTRAST_HU_MAX,
            'avg_val':CONTRAST_HU_MEAN,
            'w_center': W_CENTER_DEFAULT,
            'w_width': W_WIDTH_DEFAULT
        }
        #w
        with open(args.pkl_path, 'rb') as pkl_file:
            all_ctpes = pickle.load(pkl_file)
        #w
        self.ctpe_list = [ctpe for ctpe in all_ctpes if self._include_ctpe(ctpe)] #w 根据phase筛选样本
        self.positive_idxs = [i for i in range(len(self.ctpe_list)) if self.ctpe_list[i].is_positive]
        self.num_slices = args.num_slices
        #w
        self.window_to_series_idx = []  
        self.series_to_window_idx = []  
        window_start = 0
        for i, s in enumerate(self.ctpe_list): 
            num_windows = len(s) // self.num_slices + (1 if len(s) % self.num_slices > 0 else 0) 
            self.window_to_series_idx += num_windows * [i]   
            self.series_to_window_idx.append(window_start) 
            window_start += num_windows 
        print(len(self.window_to_series_idx))
        #sys.exit()
    #w
    def _include_ctpe(self, pe):
        if pe.phase != self.phase:
            return False
        '''if pe.is_positive and pe.type not in self.pe_types:
            return False'''
        return True
    #w
    def __len__(self):
        return len(self.window_to_series_idx)
    #w
    def __getitem__(self, idx):
        ctpe_idx = self.window_to_series_idx[idx]
        ctpe = self.ctpe_list[ctpe_idx] 
        #w 我们只选择能包含病灶切片的开始ID
        do_center_abnormality = random.random() < 0.5
        start_idx = self._get_abnormal_start_idx(ctpe, do_center = do_center_abnormality)
        #w
        if self.do_jitter:  
            start_idx += random.randint(-self.num_slices // 2, self.num_slices // 2)
            start_idx = min(max(start_idx, 0), len(ctpe) - self.num_slices)
        #w
        volume = self._load_volume(ctpe, start_idx)   
        volume = self._transform(volume)     
        #w
        target = {'is_abnormal':ctpe.is_positive,
                  'study_num':ctpe.study_num,   
                  'dset_path':str(ctpe.study_num),
                  'slice_idx':start_idx,  
                  'series_idx':ctpe_idx,
                  'bbox':ctpe.bbox}   
        #w
        return volume, target
    #w
    def get_series(self, study_num):
        for ctpe in self.ctpe_list:
            if ctpe.study_num == study_num:
                return ctpe
        return None
    #w
    def get_series_label(self, series_idx):
        series_idx = int(series_idx)
        return float(self.ctpe_list[series_idx].is_positive)
    #w
    def _get_abnormal_start_idx(self, ctpe, do_center):
        abnormal_bounds = (ctpe.first_appear, ctpe.last_appear)
        if do_center:
            center_idx = sum(abnormal_bounds) // 2
            start_idx = max(0, center_idx - self.num_slices // 2)
        else:
            #w 删掉了self.min_pe_slices的使用
            start_idx = random.randint(abnormal_bounds[0] - self.num_slices,
                                       abnormal_bounds[1] + 1)
        return start_idx
    #w
    def _load_volume(self, ctpe, start_idx):    
        with h5py.File(os.path.join(self.data_dir, 'data1.hdf5'), 'r') as hdf5_fh:     
            volume = hdf5_fh[str(ctpe.study_num)][start_idx:start_idx + self.num_slices, :, :]   
        return volume
    #w
    def _crop(self, volume, x1, y1, x2, y2):
        volume = volume[:, y1:y2, x1:x2]
        return volume
    #w
    def _rescale(self, volume, interpolation = cv2.INTER_AREA):
        return util.resize_slice_wise(volume, tuple(self.resize_shape), interpolation)
    #w
    def _pad(self, volume):
        def add_padding(volume_, pad_value = AIR_HU_VAL):
            num_pad = self.num_slices - volume_.shape[0]
            volume_ = np.pad(volume_, ((0, num_pad), (0, 0), (0, 0)), mode = 'constant', constant_values = pad_value)
            return volume_
        #w
        volume_num_slices = volume.shape[0]
        if volume_num_slices < self.num_slices:
            volume = add_padding(volume, pad_value = AIR_HU_VAL)
        elif volume_num_slices > self.num_slices:
            start_slice = (volume_num_slices - self.num_slices) // 2
            volume = volume[start_slice:start_slice + self.num_slices, :, :]
        return volume
    #w
    def _transform(self, inputs):
        inputs = self._pad(inputs)
        if self.resize_shape is not None:
            inputs = self._rescale(inputs, interpolation = cv2.INTER_AREA)  
        if self.crop_shape is not None:
            row_margin = max(0, inputs.shape[-2] - self.crop_shape[-2])    
            col_margin = max(0, inputs.shape[-1] - self.crop_shape[-1])
            #w
            row = random.randint(0, row_margin) if self.is_training_set else row_margin // 2
            col = random.randint(0, col_margin) if self.is_training_set else col_margin // 2
            inputs = self._crop(inputs, col, row, col + self.crop_shape[-1], row + self.crop_shape[-2])
        if self.do_vflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis = -2)
        if self.do_hflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis = -1)
        if self.do_rotate:
            angle = random.randint(-15, 15)
            inputs = rotate(inputs, angle, (-2, -1), reshape = False, cval = AIR_HU_VAL)
        inputs = self._normalize_raw(inputs)
        inputs = np.expand_dims(inputs, axis = 0)  
        inputs = torch.from_numpy(inputs)
        return inputs

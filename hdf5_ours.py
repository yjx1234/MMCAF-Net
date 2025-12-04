import os
import pydicom
import h5py
import numpy as np
import pandas as pd


def check_dataset_exists(hdf5_filename, dataset_name):
    try:
        with h5py.File(hdf5_filename, 'r') as f:
            # 尝试访问数据集
            f[dataset_name]  # 如果数据集不存在，这里会抛出KeyError
            return True
    except KeyError:
        # 如果数据集不存在，捕获KeyError并返回False
        return False

import pydicom
import matplotlib.pyplot as plt
dicom_file_path = '/mntcephfs/data/med/Lung-PET-CT-Dx/Lung-PET-CT-Dx/Lung_Dx-A0247/05-06-2011-NA-PETNEW03CBMWholebodyFirstHead Adult-55078/10.000000-Thorax  1.0  B70f-78680/1-001.dcm'  

ds = pydicom.dcmread(dicom_file_path)
print(ds)
sys.exit()



import sys
root_dir = '/mntcephfs/data/med/Lung-PET-CT-Dx/Lung-PET-CT-Dx'
output_h5_file = '/mntcephfs/data/med/Lung-PET-CT-Dx/temp.hdf5'
with h5py.File(output_h5_file, 'w') as h5f:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        data_list = []
        for filename in filenames:
            if filename.lower().endswith('.dcm') or filename.lower().endswith('.dicom'):
                file_path = os.path.join(dirpath, filename)
                # id=filename[:-4]
                # 使用pydicom读取DICOM文件
                ds = pydicom.dcmread(file_path)
                pixel_data = ds.pixel_array
                data_list.append(pixel_data)
                # 为每个DICOM文件在HDF5中创建一个新的数据集（你可能想要根据文件名或其他元数据命名数据集）
                # 这里我们使用简单的计数作为数据集名称，但你可能需要更复杂的命名方案
                # dataset_name = f'dataset_{len(h5f.keys())}'
        # data_list = np.stack(data_list, axis=0)
        #if dirpath[45:50]=='A0002':
        #    raise ValueError("data_list的长度为：{},data_list为{}".format(len(data_list), data_list))
        #raise ValueError("dirpath[45:50]是: {}".format(dirpath))
        print("dirpath是{}".format(dirpath))

        if 'ALPHA' in dirpath:   #去掉alpha型，因为dcm数据形状不统一，无法转成numpy数组
            continue
        # raise ValueError("file_name是: {}".format(filenames))
        if data_list==[]:
            continue
        data_list = np.array(data_list)
        if filenames==[]:
            continue
        if data_list.shape[1]!=512:
            continue
        if check_dataset_exists(output_h5_file,dirpath[45:50]):
            dataset=h5f[dirpath[45:50]]
            if dataset.shape[0]<data_list.shape[0]:
                del h5f[dirpath[45:50]]
            else:
                continue
        #print(data_list.shape[0])
        print('111', dirpath)
        #print("dirpath是{},dirnames是{},filenames是{},data_list的形状为：{}".format(dirpath,dirnames,filenames,data_list.shape))
        h5f.create_dataset(dirpath.split('/')[5].split('-')[-1], data=data_list,compression='gzip')  # 使用gzip压缩以节省空间
        #print(dirpath[45:50])
        #print(data_list.shape)


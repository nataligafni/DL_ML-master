"""
Created on Sun Nov 22 16:29:02 2020
@author: moshe
"""
import numpy as np
import os
from time import strftime


class Configuration():
    def __init__(self):
        self.data_path = r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Data'
        self.x_dir = r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Data\All'
        # self.x_dir =r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Data\Segmentation\Image'
        self.y_mask = r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Data\Segmentation\Mask'
        self.y_label_COVID = r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Data\CT_COVID'
        self.y_label_non_COVID=r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Data\CT_NonCOVID'
        self.seg_im=r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Data\Segmentation\Image'
        self.save_model = r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\weights'
        self.save_seg_results = r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Results\SegmentationResults'
        self.save_re_results = r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\Results\ReconstructionResults'
        self.run_dir = os.path.dirname(self.data_path) + '/weights/' + 'Run_at_time_' + strftime('%H-%M-%S') +\
                      '_date_' + strftime('%d-%m-%y')

        self.weights_path = r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\weights\segmentation best\weights_e0049_loss0.1127_val_loss0.2412.h5' # weights for segmentation
        # self.weights_path = r'C:\Users\dorif\PycharmProjects\Natali\DL_MTL-master\weights\classification best\weights_e0002_loss0.1928_val_loss0.2466.h5' #weights for classification


    def createdir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path





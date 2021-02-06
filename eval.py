import sys
import os
import matplotlib.pyplot as plt
from paths_and_params.Configuration import Configuration
from paths_and_params.params import Params
from sklearn.metrics import confusion_matrix
import nibabel as nib
import numpy as np
from loss_functions import dice_coef_loss
import skimage.transform
from keras import backend as K

config = Configuration()
params = Params()

def evalu(model, test_names):
    seg_output, re_output, class_output = model.run_model()
    labels = []
    Dice=[]
    for i, test_name in enumerate(test_names):
        if params.data_type == 'seg_only':
            if test_name[-3:] == 'nii' or os.path.isfile(config.y_label_non_COVID + '\\' + test_name):
                seg_result = seg_output[i]
                seg_result[seg_result < params.seg_thresh] = 0
                seg_result[seg_result >= params.seg_thresh] = 1
                if test_name[-3:] == 'nii':
                    img = nib.load(config.y_mask + '\\' + 'tr_mask_merged_' + test_name[6:])
                    y_tmp = np.array(img.dataobj)
                    y_true = np.round(skimage.transform.resize(y_tmp, params.dim))
                    intersection = np.sum(np.abs(y_true * seg_result))
                    epsilon = 1e-6
                    dice = (2 * intersection + epsilon) / (np.sum(np.abs(y_true) + np.abs(seg_result)) + epsilon)
                    Dice.append(dice)
                else:
                    y_true = np.zeros([int(params.dim[0]), int(params.dim[1]), int(params.dim[2])])


                plt.imsave(config.save_seg_results + '\\' + test_name.replace('.nii', '.png').replace('.jpg', '.png'),
                           seg_result[:, :, 0], cmap='gray')
        if class_output[i] <= params.class_thresh:
            save_re_path = config.save_re_results + '/non-covid/' + test_name.replace('.nii', '.png').replace('.jpg',
                                                                                                              '.png')
        else:
            save_re_path = config.save_re_results + '/covid/' + test_name.replace('.nii', '.png').replace('.jpg', '.png')
        plt.imsave(save_re_path, re_output[i, :, :, 0], cmap='gray')

        if test_name[-3:] == 'nii':
            y_true = 1
        else:
            if os.path.isfile(config.y_label_COVID + '\\' + str(test_name)):
                y_true = 1
            else:
                y_true = 0
        labels.append(y_true)
    predictions = [1 if n >= params.class_thresh else 0 for n in class_output]

    print('Classification Report:')
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    cm = confusion_matrix(labels, predictions)
    acc = (tp + tn) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * sensitivity * precision / (sensitivity + precision)
    print(cm)
    print('acc=%f' % acc)
    print('sensitivity=%f' % sensitivity)
    print('precision=%f' % precision)
    print('f1 = %f' % f1)

    try:
        print('Segmentation Results based on Dice:')
        print(np.mean(Dice))
    except:
        pass



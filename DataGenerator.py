
import numpy as np
from tensorflow import keras
from paths_and_params.Configuration import Configuration
import skimage.transform
import nibabel as nib
import cv2 as cv
import os
from paths_and_params.params import Params
from prepocessing.data_augmenter import data_augmentation
params = Params()
config = Configuration()


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs,batch_size=params.batch_size, dim=params.dim,
                 n_classes=2, shuffle=True, do_augs=False):
        'Initialization'
        self.dim = dim
        self.dimmask = (dim[0], dim[1], n_classes-1)
        self. n_classes= n_classes
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.x_dir = config.x_dir
        self.y_mask = config.y_mask
        self.y_label_COVID = config.y_label_COVID
        self.do_augs = do_augs
        self.size=(256,256)

        # if Test:
        #     self.test_dir = config.test_dir
        #     self.test_mask_dir = config.test_mask_dir
        #     self.label_dir = config.test_label_dir

        # self.read_labels()
        self.on_epoch_end()

    # def read_labels(self):
    #     self.labels = np.load(self.label_dir)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        Xu = np.zeros([self.batch_size, *self.dim], dtype='float16')
        # Ymask = np.zeros([self.batch_size, *self.dimmask], dtype='uint8')
        # Ylabel = np.zeros([self.batch_size, self.n_classes], dtype='uint8')
        Ymask = np.zeros([self.batch_size, *self.dimmask])
        Ylabel = np.zeros([self.batch_size, self.n_classes-1])

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if ID[-3:]=='nii':
                img = nib.load(self.x_dir + '\\' + str(ID))
                x_tmp = np.array(img.dataobj, dtype='float64')
                # x_tmp = np.rot90(x_tmp, 3)
            else:
                x_tmp = np.array(cv.imread(self.x_dir +'\\' + str(ID)))
            try:
                x=cv.cvtColor(x_tmp, cv.COLOR_BGR2GRAY)
            except:
                if x_tmp.ndim==3:
                    if x_tmp.shape[2]==3:
                        print('3D image')
                    elif x_tmp.shape[2]==1:
                        x=np.squeeze(x_tmp, axis=2).shape
                elif x_tmp.ndim == 2:
                    x = x_tmp
            x=self.resize_images(x)
            x=self.Normalization_fun(x)
            x=np.expand_dims(x, -1)
            Xu[i,] = x
            if ID[-3:]=='nii':
                # stop_ID=ID.find['.nii']
                result = ID[6:]
                new_ID='tr_mask_merged_' + result
                img = nib.load(self.y_mask +'\\'+ str(new_ID))
                y_tmp = np.array(img.dataobj)
                # y_tmp = np.rot90(y_tmp, 3)
                y = np.round(self.resize_images(y_tmp))
                y =np.expand_dims(y, -1)
                y2 =1
            else:
                # y_tmp = np.array(cv.imread(self.y_mask + '\\' + str(ID)))
                # y = np.full([int(self.size[0]),int(self.size[1]), 1], np.nan)

                if os.path.isfile(self.y_label_COVID +'\\'+ str(ID)):
                    y = -1 * np.ones([int(self.size[0]), int(self.size[1]), 1])
                    y2 = 1
                else:
                    y = np.zeros([int(self.size[0]), int(self.size[1]), 1])
                    # y = -1 * np.ones([int(self.size[0]), int(self.size[1]), 1])
                    y2 = 0
            Ymask[i,] = y
            Ylabel[i,] = y2
            # Store class
        X = Xu

        # augs
        # X,Ymask = data_augmentation(X, Ymask)
        # for i in range(0,X.shape[0]):
        #     imgae = self.Normalization_fun(X[i,])
        #     X[i,] = imgae


        y = {"seg_loss": Ymask, "re_loss": X, "class_loss": Ylabel}

        return X, y

    def resize_images(self,x):
        X_data_resized = skimage.transform.resize(x,self.size)
        return X_data_resized

    # def preprocessing(imgs, masks, name_list, Test_flag='False', save_aug_path='True'):
    def Normalization_fun(self,x):

        # global pixel standardization
        pixels = np.asarray(x)
        # convert from integers to floats
        pixels = pixels.astype('float32')
        # calculate global mean and standard deviation
        mean, std = pixels.mean(), pixels.std()
        # global standardization of pixels
        pixels = (pixels - mean) / std
        return pixels




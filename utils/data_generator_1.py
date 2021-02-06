import numpy as np
import os, cv2
import matplotlib.pyplot as plt
from data_handling.preprocessing import preprocessing
from paths_and_params.params import Params
from tensorflow.python.keras.utils.data_utils import Sequence
from utils.least_square_circle import plot_data_circle
from utils.rotate_image_and_centers import rotate_image_and_centers
params = Params()
if params.save_activation_maps:
    from utils.activation_maps import activation_maps


class data_generator(Sequence): # use keras.utils.Sequence to run with keras and not TF
    'Generates data for Keras'

    def __init__(self, list_IDs, dict_items, batch_size=32, dim=params.img_size, n_channels=1, model=None,
                 shuffle=False, train_flag=False, half_window=params.half_window1, resulted_cc=[]):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.dict_items = dict_items
        self.size_ratio_dict = list(np.load(params.project_path + '/dicts/dict_list_size_ratio_s1.npy', allow_pickle=True))
        self.train_flag = train_flag
        self.half_window = half_window
        self.resulted_cc = resulted_cc
        self.model=model
        self.on_epoch_end()

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
        #### Initialization ####
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, params.output_size))

        name_list = [None]*self.batch_size

        #### Generate data ####
        for i, ID in enumerate(list_IDs_temp):

            base_folder = params.data_path + '/'
            im_name = self.dict_items[ID]["pic_path"]

            folder_name, file_name = os.path.split(im_name)
            im_name = base_folder + file_name

            gray_img = cv2.imread(im_name,0)
            # cur_sr_dict = next(dict for dict in self.size_ratio_dict if dict['case_name'] == file_name[5:9])
            # size_ratio = cur_sr_dict['size_ratio']
            if not params.scaling_flag:
                size_ratio = 1
            half_window = int(self.half_window * size_ratio)
            half_window =  self.half_window
            if (gray_img.shape[0] < 2 * half_window) or (gray_img.shape[1] < 2 * half_window):
                raise Exception('requested window size ({}) is too large for image shape ({})'.
                                format(2 * half_window, gray_img.shape))

            #### Crop according to S1 location or according to location model run results ####
            curr_center = self.dict_items[ID]['crop_center']
            c1 = self.dict_items[ID]["circ_center1"]
            c2 = self.dict_items[ID]["circ_center2"]
            r1 = self.dict_items[ID]["r1"]
            r2 = self.dict_items[ID]["r2"]

            if not self.resulted_cc == []:
                xc, yc = curr_center[0], curr_center[1]
                col_start, row_start = np.max((xc - params.half_window1, 0)), np.max((yc - params.half_window1, 0))
                curr_center = (self.resulted_cc[i] + 1) * params.half_window1  # this needs to be half_window1, it's not a mistake
                curr_center = np.round(curr_center).astype(int) + [col_start, row_start]

            if self.train_flag & params.do_augs:
                nr = np.random.randint(low=-params.rand_range, high=params.rand_range, size=2)
                xc, yc = curr_center[0] + nr[0], curr_center[1] + nr[1]
            else:
                xc, yc = curr_center[0], curr_center[1]

            ### cropping without zero-padding ###
            if (xc - half_window) < 0:
                col_start, col_end = 0, xc + half_window + -1*(xc - half_window)
            elif (xc + half_window) > gray_img.shape[1]:
                col_start, col_end = (xc - half_window) - (xc + half_window - gray_img.shape[1]) , gray_img.shape[1]
            else:
                col_start, col_end = xc - half_window, xc + half_window

            if (yc - half_window) < 0:
                row_start, row_end = 0, yc + half_window + -1*(yc - half_window)
            elif (yc + half_window) > gray_img.shape[0]:
                row_start, row_end = (yc - half_window) - (yc + half_window - gray_img.shape[0]) , gray_img.shape[0]
            else:
                row_start, row_end = yc - half_window, yc + half_window

            ### cropping with zero-padding ###
            # col_start = np.max((xc - half_window, 0))
            # row_start = np.max((yc - half_window, 0))
            # col_end = np.min((xc + half_window, gray_img.shape[1]))
            # row_end = np.min((yc + half_window, gray_img.shape[0]))

            angle = np.random.uniform(-params.rot_range, params.rot_range)
            if self.train_flag & params.do_augs & (angle > 0):
                cropped_img, c1, c2 = rotate_image_and_centers(gray_img, angle, [xc, yc], c1, c2, half_window)
            else:
                cropped_img = gray_img[row_start:row_end, col_start:col_end]
            # cropped_img = gray_img[row_start:row_end, col_start:col_end]

            res_im = cv2.resize(cropped_img, (self.dim[0], self.dim[1]))
            res_im = np.expand_dims(res_im, 2)
            X[i] = res_im

            c1_new = c1 - [col_start, row_start]
            c2_new = c2 - [col_start, row_start]

            c1_norm = -1 + (c1_new / (2 * half_window)) * 2
            c2_norm = -1 + (c2_new / (2 * half_window)) * 2

            r1_norm = -1 + ((r1/size_ratio - 50) / 200) * 2
            r2_norm = -1 + ((r2/size_ratio - 50) / 200) * 2
            # r1_norm = -1 + ((r1 - 50) / 200) * 2
            # r2_norm = -1 + ((r2 - 50) / 200) * 2

            y[i] = [*c1_norm, r1_norm, *c2_norm, r2_norm]

            # #### Visualization for debugging #####
            # c1_unorm = (c1_norm+1)*half_window
            # r1_unorm = ((r1_norm+1)*100 +50)/size_ratio
            # plt.figure()
            # plt.subplot(1,3,1)
            # plt.imshow(np.squeeze(cropped_img), cmap='gray')
            # plt.subplot(1, 3, 2)
            # plt.imshow(np.squeeze(cropped_img), cmap='gray')
            # plot_data_circle(*c1_new, r1)
            # plt.subplot(1, 3, 3)
            # plt.imshow(np.squeeze(cropped_img), cmap='gray')
            # plot_data_circle(*c1_unorm, r1_unorm)
            # plt.show()

            name_list[i] = file_name

        X, y = preprocessing(X, y, name_list, train_flag=self.train_flag)

        if params.save_activation_maps and self.model:
                activation_maps(X, self.model, name_list, cmap='jet',activations_flag=True,heatmaps_flag=False)
        return X, y


def main():
    dict_items = list(np.load(r'../dicts/dict_annotations_clean_mean_data.npy', allow_pickle=True))
    idx = 3
    train_data = data_generator([idx], dict_items, batch_size=1, shuffle=False, train_flag=True)
    x,y = train_data[0]
    half_window = params.half_window1

    y_orig = np.array([*(y[0,0:2] + 1)*half_window, (y[0,2] + 1)*100+50,
                      *(y[0,3:5] + 1)*half_window, (y[0,-1] + 1)*100+50])
    y = np.squeeze(y)
    print(y)
    print(y_orig)
    scale = half_window*2 / params.img_size[0]
    res_im = cv2.resize(np.squeeze(x), (int(params.img_size[0]*scale), int(params.img_size[1]*scale)))
    plt.imshow(res_im, cmap='gray')
    plot_data_circle(*y_orig[0:3])
    plot_data_circle(*y_orig[3:])
    plt.show()


if __name__ == '__main__':
    main()

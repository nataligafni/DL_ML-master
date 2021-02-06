import numpy as np
from paths_and_params.Configuration import Configuration
import os
from sklearn.model_selection import GroupShuffleSplit

config=Configuration()

seg_path = config.seg_im
os.chdir(seg_path)
seg_im = os.listdir()

class_non_path = config.y_label_non_COVID
os.chdir(class_non_path)
class_non_covid = os.listdir()

class_covid_path = config.y_label_COVID
os.chdir(class_covid_path)
class_covid = os.listdir()

train_seg, valid_seg = next(GroupShuffleSplit(test_size=.15, n_splits=2, random_state=7).split(seg_im, groups=seg_im))
train = np.array(seg_im)[train_seg]
train_seg, test_seg = next(GroupShuffleSplit(test_size=.2, n_splits=2, random_state=7).split(train, groups=train))

train_class_non, valid_class_non = next(
    GroupShuffleSplit(test_size=.15, n_splits=2, random_state=7).split(class_non_covid, groups=class_non_covid))
train_class_n = np.array(class_non_covid)[train_class_non]
train_class_non, test_class_non = next(
    GroupShuffleSplit(test_size=.2, n_splits=2, random_state=7).split(train_class_n, groups=train_class_n))

train_class_covid, valid_class_covid = next(
    GroupShuffleSplit(test_size=.15, n_splits=2, random_state=7).split(class_covid, groups=class_covid))
train_class_y = np.array(class_covid)[train_class_covid]
train_class_covid, test_class_covid = next(
    GroupShuffleSplit(test_size=.2, n_splits=2, random_state=7).split(train_class_y, groups=train_class_y))

train_names = []
validation_names = []
test_names = []

for index in train_seg:
    train_names.append(train[index])
for index in train_class_non:
    train_names.append(train_class_n[index])
for index in train_class_covid:
    train_names.append(train_class_y[index])
for index in valid_seg:
    validation_names.append(seg_im[index])
for index in test_class_non:
    validation_names.append(class_non_covid[index])
for index in test_class_covid:
    validation_names.append(class_covid[index])
for index in test_seg:
    test_names.append(train[index])
for index in test_class_non:
    test_names.append(train_class_n[index])
for index in test_class_covid:
    test_names.append(train_class_y[index])

names = np.array([train_names, validation_names, test_names])
np.save(config.data_path+'/names.npy', names)

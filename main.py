import sys
import os


import matplotlib.pyplot as plt
import numpy as np
from MultiTaskModel import MultiTaskModel
from paths_and_params.Configuration import Configuration
from DataGenerator import DataGenerator
from paths_and_params.params import Params
from eval import evalu



config = Configuration()
params = Params()

def generating_validation_plots(output):
    # plot the total loss, classification loss, reconstruction loss, segmentation loss
    lossNames = ["loss", "class_loss_loss", "re_loss_loss", "seg_loss_loss"]
    titles = ["Total Loss", "Classification Loss", "Reconstruction Loss", "Segmentation Loss"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(4, 1, figsize=(13, 13))
    # loop over the loss names
    for (i, l) in enumerate(lossNames):
        # plot the loss for both the training and validation data
        title = "{}".format(titles[i])
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Loss")
        ax[i].plot(np.arange(0, len(output.history['loss'])), output.history[l], label=l)
        ax[i].plot(np.arange(0, len(output.history['loss'])), output.history["val_" + l],
                   label="val_" + l)
        ax[i].legend()
    # save the losses figure
    plt.tight_layout()
    plt.show()
    # plt.savefig("{}_losses.png".format(args["plot"]))
    # plt.close()



def seg_only(names):
    seg_names = []
    for i, name in enumerate(names):
        if name[-3:]=='nii':
            seg_names.append(name)
        if os.path.isfile(config.y_label_non_COVID +'\\'+ name) and i%4==0:
            seg_names.append(name)

    return seg_names

def not_seg(names):
    not_nii_names = []
    for i, name in enumerate(names):
        if name[-3:]!='nii':
            not_nii_names.append(name)
            # if os.path.isfile(config.y_label_non_COVID +'\\'+ name) and i%2!=0:
            #     not_nii_names.append(name)
            # if os.path.isfile(config.y_label_COVID +'\\'+ name):
            #     not_nii_names.append(name)

    return not_nii_names


if __name__ == '__main__':
    #initialization


    train_names, validation_names, test_names = np.load(config.data_path+'/names.npy', allow_pickle=True)

    if params.data_type == 'seg_only':
        train_names, validation_names, test_names = seg_only(train_names), seg_only(validation_names), \
                                                    seg_only(test_names)
    elif params.data_type == 'not_seg':
        train_names, validation_names, test_names = not_seg(train_names), not_seg(validation_names),\
                                                    not_seg(test_names)

    Training_generator = DataGenerator(train_names, do_augs=True)
    Validation_generator = DataGenerator(validation_names)
    Test_generator = DataGenerator(test_names, shuffle=False, batch_size=1)

    #Train

    # model = MultiTaskModel(params.epochs, Training_generator, Validation_generator, Test_generator, 'train')
    # output = model.run_model
    # generating_validation_plots(output)

    #Test and Evaluation
    model = MultiTaskModel(params.epochs, Training_generator, Validation_generator, Test_generator, 'test')
    evalu(model, test_names)


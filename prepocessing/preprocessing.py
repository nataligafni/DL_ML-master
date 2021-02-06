from paths_and_params.Configuration import Configuration
from numpy import asarray


# def preprocessing(imgs, masks, name_list, Test_flag='False', save_aug_path='True'):
def preprocessing(imgs):

    config=Configuration

    # global pixel standardization
    # load image
    pixels = asarray(imgs)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate global mean and standard deviation
    mean, std = pixels.mean(), pixels.std()
    # global standardization of pixels
    pixels = (pixels - mean) / std

    # if Test_flag=='False':

        # aug = Data_augmentation._to_deterministic()
        # new_img = aug.augment_image(img)
        # new_mask = aug.augment_image(mask)



        # if save_aug_path:
        #     # try: os.mkdir(os.path.join(params.save_aug_path, "saved_augs"))
        #     # except: pass
        #     for i,img in enumerate(imgs):
        #         im_path = config.save_augmentation + '/augmented_' + str(name_list[i])
        #         plt.imsave(im_path,im)


    return imgs






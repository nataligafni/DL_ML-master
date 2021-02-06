import numpy as np
from imgaug import augmenters as iaa

#######################   Augmentations   ####################################
def data_augmentation(batch_imgs, batch_masks):
    # batch_imgs = np.float32(batch_imgs)
    seq_images_only = iaa.Sequential([
        iaa.SomeOf((0, 1),[
            iaa.GaussianBlur((0.1, 2.0)),
            # iaa.AverageBlur(k=(3, 7)),
            # iaa.MedianBlur(k=(3, 7)),
        ]),

        iaa.SomeOf((0, 1),[
            # iaa.LogContrast(gain=[0.1, 0.3, 0.5, 0.7, 0.9]),
            # iaa.SigmoidContrast(gain=[5]),
            # iaa.GammaContrast(gamma=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5]),
            iaa.Multiply((0.5, 1.5)),
            # iaa.Add((-5/255, 5/255))
        ])
    ])

    #
    # seq_masks_and_images = iaa.Sequential([
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    # ])

    [p1,p2] = np.random.uniform(size=2)

    if p1 <= 0.7:
        batch_imgs = seq_images_only.augment_images(batch_imgs)

    # if p2 <= 0.7:
    #     seq_det = seq_masks_and_images.to_deterministic()
    #     batch_imgs = seq_det.augment_images(batch_imgs)
    #     batch_masks = seq_det.augment_images(batch_masks)


    return batch_imgs, batch_masks

#####################################################################################
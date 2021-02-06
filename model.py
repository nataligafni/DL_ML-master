from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from loss_functions import dice_coef_loss

from paths_and_params.params import Params
params = Params()


def network_model():
    inputs = Input(params.dim)
    conv1 = BatchNormalization()(inputs)
    conv1 = Conv2D(params.feature_maps, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    conv1 = Conv2D(params.feature_maps, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=params.pool_size)(conv1)
    conv2 = BatchNormalization()(pool1)
    conv2 = Conv2D(params.feature_maps * 2, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    conv2 = Conv2D(params.feature_maps * 2, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=params.pool_size)(conv2)
    conv3 = BatchNormalization()(pool2)
    conv3 = Conv2D(params.feature_maps * 4, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(params.feature_maps * 4, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=params.pool_size)(conv3)
    conv4 = BatchNormalization()(pool3)
    conv4 = Conv2D(params.feature_maps * 8, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = Conv2D(params.feature_maps * 8, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(params.drop_out)(conv4)
    pool4 = MaxPooling2D(pool_size=params.pool_size)(drop4)

    conv5 = BatchNormalization()(pool4)
    conv5 = Conv2D(params.feature_maps * 16, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    conv5 = Conv2D(params.feature_maps * 16, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(params.drop_out)(conv5)

    ## Segmentation task
    up6 = Conv2D(params.feature_maps * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=params.up_sample_size)(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(params.feature_maps * 8, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(params.feature_maps * 8, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(params.feature_maps * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=params.up_sample_size)(conv6))
    up7 = Dropout(params.drop_out)(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(params.feature_maps * 4, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(params.feature_maps * 4, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(params.feature_maps * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=params.up_sample_size)(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(params.feature_maps * 2, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(params.feature_maps * 2, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(params.feature_maps, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=params.up_sample_size)(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(params.feature_maps, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(params.feature_maps, params.kernel, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    seg_output = Conv2D(1, 1, activation='sigmoid', name="seg_loss")(conv9)  # output of segmentation task

    ## Reconstruction task
    r6 = Conv2D(params.feature_maps * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=params.up_sample_size)(drop5))
    merge_re6 = concatenate([drop4, r6], axis=3)
    conv_re6 = Conv2D(params.feature_maps * 8, params.kernel, activation='relu', padding='same',
                      kernel_initializer='he_normal')(merge_re6)
    conv_re6 = Conv2D(params.feature_maps * 8, params.kernel, activation='relu', padding='same',
                      kernel_initializer='he_normal')(conv_re6)

    r7 = Conv2D(params.feature_maps * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=params.up_sample_size)(conv_re6))
    r7 = Dropout(params.drop_out)(r7)
    merge_re7 = concatenate([conv3, r7], axis=3)
    conv_re7 = Conv2D(params.feature_maps * 4, params.kernel, activation='relu', padding='same',
                      kernel_initializer='he_normal')(merge_re7)
    conv_re7 = Conv2D(params.feature_maps * 4, params.kernel, activation='relu', padding='same',
                      kernel_initializer='he_normal')(conv_re7)

    r8 = Conv2D(params.feature_maps * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=params.up_sample_size)(conv_re7))
    merge_re8 = concatenate([conv2, r8], axis=3)
    conv_re8 = Conv2D(params.feature_maps * 2, params.kernel, activation='relu', padding='same',
                      kernel_initializer='he_normal')(merge_re8)
    conv_re8 = Conv2D(params.feature_maps * 2, params.kernel, activation='relu', padding='same',
                      kernel_initializer='he_normal')(conv_re8)

    r9 = Conv2D(params.feature_maps, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=params.up_sample_size)(conv_re8))
    merge_re9 = concatenate([conv1, r9], axis=3)
    conv_re9 = Conv2D(params.feature_maps, params.kernel, activation='relu', padding='same',
                      kernel_initializer='he_normal')(merge_re9)
    conv_re9 = Conv2D(params.feature_maps, params.kernel, activation='relu', padding='same',
                      kernel_initializer='he_normal')(conv_re9)
    conv_re9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_re9)
    re_output = Conv2D(1, 1, activation='linear', name="re_loss")(conv_re9)  # output of reconstruction task

    ## Classification task
    cl6 = Conv2D(params.feature_maps * 2, params.kernel, activation='relu', padding='same',
                 kernel_initializer='he_normal')(drop5)
    pool_cl6 = MaxPooling2D(pool_size=params.pool_size)(cl6)
    flat = Flatten()(pool_cl6)
    cl7 = Dense(params.feature_maps * 2, activation='elu')(flat)
    drop_cl1 = Dropout(params.drop_out/2)(cl7)
    cl8 = Dense(params.feature_maps, activation='elu')(drop_cl1)
    drop_cl2 = Dropout(params.drop_out/5)(cl8)
    class_output = Dense(1, activation='sigmoid', name="class_loss")(drop_cl2)

    model = Model(inputs, [seg_output, re_output, class_output])

    losses = {"seg_loss": dice_coef_loss, "re_loss": 'mse', "class_loss": 'binary_crossentropy'}

    lossWeights = {"seg_loss": params.loss_weights[0], "re_loss": params.loss_weights[1], "class_loss": params.loss_weights[2]}
    model.compile(optimizer=Adam(lr=params.lr), loss=losses,
                  loss_weights=lossWeights)

    return model


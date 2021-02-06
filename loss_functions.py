from keras import backend as K
import tensorflow as tf


def dice_coef(y_true, y_pred, epsilon=1e-6):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    """

    def dice_func():
        intersection = K.sum(K.abs(y_true * y_pred))
        return (2 * intersection + epsilon) / (K.sum(K.abs(y_true) + K.abs(y_pred)) + epsilon)
    def no_seg():
        return tf.constant(1,dtype='float32')

    return tf.cond(tf.less(tf.reduce_sum(y_true),-1), no_seg,dice_func)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)





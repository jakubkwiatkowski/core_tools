from functools import partial
from core_tools import ops as K
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, binary_crossentropy, \
    kl_divergence

BinaryCrossEntropy = partial(BinaryCrossentropy, from_logits=True)
CrossEntropy = partial(SparseCategoricalCrossentropy, from_logits=True)
binary_cross_entropy = partial(binary_crossentropy, from_logits=True)


def kl_div(y_true, y_pred, from_logits=True, sparse=True, sigmoid="auto", expand_sigmoid=True):
    if sigmoid == "auto":
        sigmoid = True if y_pred.shape[-1] == 1 else False
    if from_logits:
        if sigmoid:
            y_pred = tf.nn.sigmoid(y_pred)
        else:
            y_pred = tf.nn.softmax(y_pred)
    if sparse == "logits":
        if sigmoid:
            y_true = tf.nn.sigmoid(y_true)
        else:
            y_true = tf.nn.softmax(y_true)
    elif sparse:
        # Older TensorFlow version do not support int8 in one-hot.
        y_true = tf.one_hot(K.uint(y_true), depth=y_pred.shape[-1])
    if sigmoid is True and expand_sigmoid:
        y_pred = tf.concat([1 - y_pred, y_pred], axis=-1)
        y_true = tf.concat([1 - y_true, y_true], axis=-1)
    # print(f"kl - {y_true} {y_pred}")
    result = kl_divergence(y_true, y_pred)
    if sigmoid is True:
        result = result[..., None]
    return result


def symmetrical_divergence(y_true, y_pred, *args, fn=kl_div, **kwargs):
    return (fn(y_true, y_pred, *args, **kwargs) + fn(y_pred, y_true, *args, **kwargs)) / 2


sym_div = symmetrical_divergence

# jensen_shannon_divergence
# probably works only for probabilities
def js_div(y_true, y_pred, from_logits=True, sparse=True, sigmoid="auto", expand_sigmoid=True):
    return kl_div(y_true, (y_true + y_pred) / 2, from_logits=from_logits, sparse=sparse, sigmoid=sigmoid,
                  expand_sigmoid=expand_sigmoid) + \
        kl_div(y_pred, (y_true + y_pred) / 2, from_logits=from_logits,
               sparse=sparse, sigmoid=sigmoid, expand_sigmoid=expand_sigmoid) / 2


# jensen_shannon_divergence 2
def js_div_2(y_true, y_pred, from_logits=True, sparse=True, sigmoid="auto", expand_sigmoid=True):
    if sigmoid == "auto":
        sigmoid = True if y_pred.shape[-1] == 1 else False
    if from_logits:
        if sigmoid:
            y_pred = tf.nn.sigmoid(y_pred)
        else:
            y_pred = tf.nn.softmax(y_pred)
    if sparse == "logits":
        if sigmoid:
            y_true = tf.nn.sigmoid(y_true)
        else:
            y_true = tf.nn.softmax(y_true)
    elif sparse:
        # Older TensorFlow version do not support int8 in one-hot.
        y_true = tf.one_hot(K.uint(y_true), depth=y_pred.shape[-1])
    if sigmoid is True and expand_sigmoid:
        y_pred = tf.concat([1 - y_pred, y_pred], axis=-1)
        y_true = tf.concat([1 - y_true, y_true], axis=-1)
    # print(f"kl - {y_true} {y_pred}")
    result = kl_divergence(y_true, (y_true + y_pred) / 2) + kl_divergence(y_pred, (y_true + y_pred) / 2)
    if sigmoid is True:
        result = result[..., None]
    return result


def cross_entropy_with_logits(y_true, y_pred, axis=-1, true_from_logits=True, sigmoid=False):
    if true_from_logits:
        if sigmoid:
            y_true = tf.nn.sigmoid(y_true)
        else:
            y_true = tf.nn.softmax(y_true, axis=axis)
    # print(f"cross_entropy - {y_true} {y_pred}")
    if sigmoid:
        return tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred, axis=axis)


def debug_(fn):
    def wrapper(y_true, y_pred):
        # print(f"{fn} - {y_true} {y_pred}")
        result = fn(y_true, y_pred)
        return result

    return wrapper

# y_true.numpy().sum(1)
# tf.nn.sigmoid(y_pred).numpy().sum(1)
# y_true


# def sigmoid_kl_divergence(y_true, y_pred):
#     return kl_divergence(y_true, tf.nn.sigmoid(y_pred))

# y_pred.numpy()
# y_true.numpy()
# tf.nn.softmax(y_pred).numpy()
# y_true.numpy()
# tf.one_hot(y_true, depth=y_pred.shape[-1]).numpy()
# kl_divergence(y_true, y_pred).numpy()
# tf.concat([1 - y_true, y_true], axis=-1).numpy()

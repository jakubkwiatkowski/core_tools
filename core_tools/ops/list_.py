import tensorflow as tf
from core_tools.core import il, lw
from tensorflow.keras.layers import Layer
import math
from functools import partial

from core_tools import ops as K
from core_tools.ops import take


class DivideDim(Layer):
    def __init__(self, model, axis=1, root=2, row=None, replace_first=-1):
        super().__init__()
        self.model = model
        self.axis = axis
        self.root = root
        self.row = row
        self.replace_first = replace_first

    def build(self, input_shape):
        self.shape = tuple(input_shape)
        if self.row is None:
            self.row = tuple(
                [math.floor(self.shape[self.axis] ** (1 / self.root))] * self.root
            )
        # todo Only for 2 dim.
        elif isinstance(self.row, int):
            self.row = (
                self.row,
                math.floor(self.shape[self.axis] / self.row)
            )

        self.newshape = self.shape[:self.axis] + self.row + self.shape[self.axis + 1:]
        if self.replace_first is not False:
            self.shape = (self.replace_first,) + self.shape[1:]
            self.newshape = (self.replace_first,) + self.newshape[1:]

    def call(self, inputs):
        x = tf.reshape(inputs, self.newshape)
        x = self.model(x)
        x = tf.reshape(x, self.shape)
        return x


def shuffle(*value, axis=1, seed=None, name=None):
    indexes = tf.random.shuffle(tf.range(tf.shape(value[0])[axis]), seed=seed, name=name)
    result = []
    for v in value:
        result.append(v[(slice(None),) * (axis) + (indexes,)])
    return result


def transpose(*value, axis=(1, 0)):
    if not il(axis):
        axis = tuple(axis)
    result = []
    for v in value:
        result.append(K.transpose(v, axis=axis))
    return result


# def shuffle(*value, axis=1, seed=None, name=None):
#     indexes = tf.random.shuffle(tf.range(tf.shape(value[0])[axis]), seed=seed, name=name)
#     result = []
#     for v in value:
#         result.append(v[(slice(None),) * (axis) + (indexes,)])
#     return result
#
#
# def transpose(*value, axis=(1, 0)):
#     if not il(axis):
#         axis = tuple(axis)
#     result = []
#     for v in value:
#         result.append(K.transpose(v, axis=axis))
#     return result


def run(fn, x):
    result = []
    for i in range(len(lw(x)[0])):
        result.append(fn(take(x, i)))
    res = []
    for j in range(len(result[0])):
        res.append(tf.stack(take(result, j)))
    return res

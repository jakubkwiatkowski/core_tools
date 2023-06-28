import tensorflow as tf

from core_tools import ops as K
from core_tools.ops import init


def switch(*args, layers, prob=None, vec=True):
    if vec is True:
        vec = tf.vectorized_map
    elif not callable(vec):
        def _run(fn, data):
            return fn(data)

        vec = _run
    if prob:
        mul = 1 / prob
    else:
        mul = 1

    index = init.var(
        () if not callable(vec) else tf.shape(args[0])[0:1],
        max=len(layers * mul),
        mode="int"
    )

    def single_switch(x):
        return K.switch(x[0], *x[1:], layers=layers)

    return vec(single_switch, [index, *args])


# def switch_rand_single(inputs, layers, prob=None):
#     if prob:
#         mul = 1 / prob
#     else:
#         mul = 1
#     index = init.var(
#         (), max=len(layers * mul), mode="int"
#     )
#     return switch(
#         inputs=inputs,
#         index=index,
#         layers=layers
#     )
def categorical_like(inputs, prob=0.5, from_logits=False, dtype="int32"):
    if isinstance(prob, float):
        prob = [[1 - prob, prob]]
    if from_logits is False:
        prob = tf.math.log(prob)
    return tf.cast(tf.random.categorical(prob, num_samples=tf.shape(inputs)[0]), dtype=dtype)

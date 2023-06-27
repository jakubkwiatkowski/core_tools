import tensorflow as tf


def mask_update(x, mask, model):
    return tf.tensor_scatter_nd_update(x, tf.range(tf.shape(x)[0])[mask][..., None], model(x[mask]))


def categorical_like(inputs, prob=0.5, from_logits=False, dtype="int32"):
    if isinstance(prob, float):
        prob = [[1 - prob, prob]]
    if from_logits is False:
        prob = tf.math.log(prob)
    return tf.cast(tf.random.categorical(prob, num_samples=tf.shape(inputs)[0]), dtype=dtype)


def random_update(x, model, prob=0.5):
    mask = categorical_like(x, prob=prob, dtype=tf.bool)[0]
    return mask_update(x, mask, model)

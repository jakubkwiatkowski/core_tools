# https://keras.io/examples/vision/semisupervised_simclr/#architecture
import tensorflow as tf
from core_tools.core import is_num
from tensorflow import keras



# Define the contrastive model with model-subclassing
class SimCLR(keras.Model):
    def __init__(self, sim=0.1):
        super().__init__()
        if is_num(sim):
            self.sim = CosineSimilarity(sim)
        else:
            self.sim = sim

    def call(self, inputs):
        projections_1 = inputs[0]
        projections_2 = inputs[1]

        similarities = self.sim([projections_1, projections_2])

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2


class CosineSimilarity(keras.layers.Layer):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def call(self, inputs):
        projections_1 = inputs[0]
        projections_2 = inputs[1]

        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities = (
                tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )
        return similarities


# version of mse with sum instad of mean
def sum_squared_error(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_sum(tf.math.squared_difference(y_pred, y_true), axis=-1)


class TripleLoss(keras.layers.Layer):
    def __init__(self, margin=0.5, sim=None):
        super().__init__()
        self.sim = sim % sum_squared_error

    def call(self, inputs):
        anchor = inputs[0]
        positive = inputs[1]
        negative = inputs[2]

        ap_sim = self.sim(anchor, positive)
        an_sim = self.sim(anchor, negative)

        loss = ap_sim - an_sim
        loss = tf.maximum(loss + self.margin, 0.0)

        return loss

# from tensorflow.keras.losses import mse
# from tensorflow.math import squared_difference
# from tensorflow_addons.losses import contrastive_loss, npairs_loss

# triplet_loss = tf.math.log1p(tf.math.exp(hard_positives - hard_negatives))

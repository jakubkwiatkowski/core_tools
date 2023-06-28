import tensorflow as tf

from core_tools.core import filter_keys

# works only for Adam and RMSPROP

def tmp_compile(model, *args, **kwargs):
    ERROR_KERAS_OPTIMIZATION = "You are trying to restore a checkpoint from a legacy Keras optimizer into a v2.11+ Optimizer, which can cause errors. Please update the optimizer referenced in your code to be an instance of `tf.keras.optimizers.legacy.Optimizer`, e.g.: `tf.keras.optimizers.legacy.Adam`."
    ERROR_KERAS_OPTIMIZATION_2 = "You are trying to restore a checkpoint from a legacy Keras optimizer into a v2.11+ Optimizer, which can cause errors. Please update the optimizer referenced in your code to be an instance of `tf.keras.optimizers.legacy.Optimizer`, e.g.: `tf.keras.optimizers.legacy.RMSprop`."

    try:
        return model.compile(*args, **kwargs)
    except ValueError as e:
        if e.args[0] == ERROR_KERAS_OPTIMIZATION or e.args[0] == ERROR_KERAS_OPTIMIZATION_2:
            kwargs = filter_keys(kwargs, ["optimizer"], reverse=True)
            return model.compile(tf.keras.optimizers.legacy.Adam(), **kwargs)
        else:
            raise e

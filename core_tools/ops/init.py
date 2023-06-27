import tensorflow.experimental.numpy as np

from functools import partial


def var(shape=(10, 64, 64, 3), mode="uniform", min=0, max=1, dtype=None):
    if mode == "normal" or mode == "n":
        return np.random.normal(min, max, shape)
    elif mode == "zero" or mode == 0:
        return np.zeros(shape)
    elif mode == "one" or mode == 1:
        return np.ones(shape)
    elif mode == "int" or isinstance(mode, int):
        return np.random.randint(min, max, shape, dtype=dtype if dtype else "int32")
    return np.random.uniform(min, max, shape)


def image(shape=(10, 224, 224, 3), pre=False, mode=None, min=None, max=None):
    if isinstance(shape, int):
        shape = (10, shape, shape, 1)
    if pre:
        mode = mode or "int"
        min = min or 0
        max = max or 256
    else:
        mode = mode or "uniform"
        min = min or 0
        max = max or 1
    return var(
        shape=shape,
        mode=mode,
        min=min,
        max=max
    )


# init_raw = partial(init_image, mode="int")
vec = partial(var, shape=(32, 64))
label = partial(var, max=10, mode="int", shape=(32, 64))

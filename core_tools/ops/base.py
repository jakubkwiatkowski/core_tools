import copy
import tensorflow as tf
from functools import partial, wraps
from funcy import rcompose

pylen = len
py_int = int
py_float = float

def il(data):
    return isinstance(data, (list, tuple))

def lw(data, none_empty="remove_none", convert_tuple=True):
    if isinstance(data, list):
        if none_empty == "remove_none":
            return [d for d in data if d is not None]
        return data
    elif isinstance(data, tuple) and convert_tuple:
        return list(data)
    if none_empty and data is None:
        return []
    return [data]


def lu(data):
    if hasattr(data, "__len__") and len(data) == 1:
        return data[0]
    return data

def extract_args(args):
    if il(args):
        if il(args[0]):
            return tuple(args[0])
        return tuple(args)
    return tuple(lw(args))
def comp(*args):
    return rcompose(*extract_args(args))


int8 = partial(tf.cast, dtype="int8")
int16 = partial(tf.cast, dtype="int16")
int32 = partial(tf.cast, dtype="int32")
int64 = partial(tf.cast, dtype="int64")

uint8 = partial(tf.cast, dtype="uint8")
uint = uint8

float32 = partial(tf.cast, dtype="float32")
float64 = partial(tf.cast, dtype="float64")
bool = partial(tf.cast, dtype="bool")


int = int32
float = float32

sum = comp([partial(tf.cast, dtype=tf.float32), partial(tf.reduce_sum, axis=-1)])
mean = comp([partial(tf.cast, dtype=tf.float32), partial(tf.reduce_mean, axis=-1)])
any = comp([partial(tf.cast, dtype=tf.bool), partial(tf.reduce_any, axis=-1)])
all = comp([partial(tf.cast, dtype=tf.bool), partial(tf.reduce_all, axis=-1)])

def take(a, index=0):
    return [x[index] for x in a]


def vec(fn, *args, **kwargs):
    if not args or not kwargs:
        fn = copy.deepcopy(partial(fn, *args, **kwargs))

    def wrapper(inputs):
        # tf.print(f"Run: {fn.keywords}")
        return tf.vectorized_map(fn, inputs)

    return wrapper


def shuffle(value, axis=1, seed=None, name=None):
    perm = list(range(len(tuple(value.shape))))
    perm[axis], perm[0] = perm[0], perm[axis]
    value = tf.random.shuffle(tf.transpose(value, perm=perm), seed=seed, name=name)
    value = tf.transpose(value, perm=perm)
    return value


def map_batch(x, fn):
    shape = tf.shape(x)
    reshaped = tf.reshape(x, shape=tf.concat([tf.reduce_prod(shape[:2], keepdims=True), shape[2:]], axis=0))
    result = fn(reshaped)
    output = []
    for r in lw(result):
        output.append(tf.reshape(r, tf.concat([shape[:2], tf.shape(r)[1:]], axis=-1)))
    return lu(output)


def batch(x, wrap=False):
    shape = tf.shape(x)
    b = shape[0] if pylen(x.shape) else 1
    if wrap:
        return [b]
    return b


def gather(a, index):
    newaxis = [slice(None)] + [None] * (len(tf.convert_to_tensor(index).shape) - 1)
    return a[tf.range(batch(index))[tuple(newaxis)], index]

def create_image_grid2(x, col, row):
    shape = tf.shape(x)
    x = tf.reshape(x, tf.stack([shape[0], int(x.shape[1] / (col * row)), col, row] + list(shape[-3:])))

    shape = x.shape
    x = tf.transpose(x, (0, 1, 2, 4, 3, 5, 6))
    x = tf.reshape(x, tf.stack([tf.shape(x)[0], shape[1], shape[2] * shape[4], shape[3] * shape[5], shape[6]]))
    return x

def apply_first(fn, fn2):
    # @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(fn2(args[0]), *args[1:], **kwargs)

    return wrapper

def apply(fn, fn2):
    @wraps(fn)
    def wrapper(x, *args, **kwargs):
        return fn(lu([fn2(a) for a in lw(x)]), *args, **kwargs)

    return wrapper
def change_type(a):
    result = [a[0]]
    for i in a[1:]:
        result.append(tf.cast(i, dtype=a[0].dtype))
    return result

cat = apply(apply_first(partial(tf.concat, axis=-1), change_type), tf.convert_to_tensor)

import collections
import dataclasses
import functools
import inspect
import operator
import itertools
from functools import partial
from typing import Union, re, Callable, Dict

import numpy as np
from funcy import rcompose, identity
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.applications import MobileNetV3Small, VGG16, VGG19
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Layer, Conv2D, Dense, Reshape
from loguru import logger
from huggingface_hub import KerasModelHubMixin

from tensorflow.keras.layers import Flatten, MaxPooling2D, AveragePooling1D, GlobalMaxPooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D, Lambda, BatchNormalization, LayerNormalization

from core_tools.flatten2d import Flatten2D

MP = "MP"
AP = "AP"
GM = "GM"
GA = "GA"
GM1 = "GM1"
GA1 = "GA1"
FIRST = "first"
LAST = "last"
FLATTEN = "flatten"
FLAT = "flat"
BN = "BN"
LN = "LN"

INPUTS = "inputs"
INPUT = "input"
TARGET = "target"
MASK = "mask"
INDEX = "index"
PREDICT = "predict"
LABELS = "labels"
IMAGE = "image"
IMAGES = "images"
OUTPUT = "output"
OUTPUTS = "outputs"
INITIAL = "initial"
LATENTS = "latents"
RECONSTRUCTION = "reconstruction"
LOSS = "loss"
ATTENTION = "attention"
METRIC = "metric"
METRICS = "metrics"
AUTO = "auto"
AUGMENTATION = "augmentation"
CROSS_ENTROPY = "cross_entropy"
ACC = "acc"

MODEL_DEPTH = "model_depth"
GLOBAL_STORAGE = {
    MODEL_DEPTH: 0,
    "flow": []
}
def is_int(obj, bool_=False):
    if not bool_ and isinstance(obj, bool):
        return False
    return isinstance(obj, (int, np.integer))
    # return issubclass(type(obj),int)


def is_float(obj):
    return isinstance(obj, (float, np.floating))


def is_num(obj, bool_=False):
    return is_int(obj, bool_) or is_float(obj)


def il(data):
    return isinstance(data, (list, tuple))


def extract_args(args):
    if il(args):
        if il(args[0]):
            return tuple(args[0])
        return tuple(args)
    return tuple(lw(args))


def interleave(lists):
    return [val for tup in itertools.zip_longest(*lists) for val in tup]


def dict_from_list(keys=None, values=None) -> Dict:
    if values is None:
        values = list(range(len(lw(keys))))
    elif callable(values):
        values = [values(k) for k in keys]
    if keys is None:
        keys = list(range(len(lw(values))))
    elif callable(keys):
        keys = [keys(v) for v in values]
    if not isinstance(values, dict):
        values = dict(zip(keys[:len(lw(values))], lw(values)))
    return values


def dict_from_list2(keys, values=None):
    if values is None:
        values = list(range(len(keys)))
    return dict(zip(keys[:len(lw(values))], lw(values)))


def flatten_return(*args):
    result = []
    for arg in args:
        result.extend(lw(arg, convert_tuple=True))
    return lu(tuple(result))


def append_global_flow(x):
    GLOBAL_STORAGE["flow"].append(x)


class OverrideDict(dict):
    def to_dict(self):
        # return {k: v for k, v in self.items()}
        return dict(self)

    def get_fn(self, index):
        return self.__getitem__(index)


class PassDict(OverrideDict):
    def __missing__(self, key):
        return key


class CalcDict(OverrideDict):
    def operate(self, other, op, right=False):
        if not hasattr(other, "__iter__"):
            other = [other] * len(self)
        if is_dict(other):
            other = other.values()
        it = iter(other)
        if right:
            fn = lambda x, y: op(y, x)
        else:
            fn = op
        return CalcDict({k: fn(v, next(it)) for k, v in self.items()})

    def __add__(self, other):
        return self.operate(other, operator.add)

    def __sub__(self, other):
        return self.operate(other, operator.sub)

    def __mul__(self, other):
        return self.operate(other, operator.mul)

    def __truediv__(self, other):
        return self.operate(other, operator.truediv)

    def __radd__(self, other):
        return self.operate(other, operator.add, right=True)

    def __rsub__(self, other):
        return self.operate(other, operator.sub, right=True)

    def __rmul__(self, other):
        return self.operate(other, operator.mul, right=True)

    def __rtruediv__(self, other):
        return self.operate(other, operator.truediv, right=True)


def get_init(obj):
    if inspect.isclass(obj):
        return inspect.signature(obj.__init__)
    return inspect.signature(obj)


def get_init_dict(obj):
    return get_init(obj).parameters


def get_init_args(obj):
    return list(get_init_dict(obj).keys())


def get_args(obj):
    try:
        return list(inspect.signature(obj).parameters.keys())
    except Exception as e:
        logger.error(f"{e}. Returning 1.")
        return ["arg"]


def comp(*args):
    return rcompose(*extract_args(args))


def call(fn, x=None):
    if len(get_args(fn)) > 0:
        if isinstance(x, dict):
            return fn(**x)
        elif il(x):
            return fn(*x)
        else:
            return fn(x)
    else:
        return fn()


def get_str_comp(str_comp):
    if callable(str_comp):
        return str_comp
    elif isinstance(str_comp, str):
        if str_comp == "in":
            return lambda x, y: x in y
        elif str_comp in ["equal", "="]:
            return lambda x, y: x == y
        elif str_comp == "re":
            return lambda x, y: re.match(x, y)
        elif str_comp == "start":
            return lambda x, y: y.startswith(x)
        elif str_comp == "end":
            return lambda x, y: y.endswith(x)

        return lambda x, y: x == y


def test(filters, x=None, str_comp="in"):
    str_comp = get_str_comp(str_comp)
    result = []
    for f in lw(filters):
        if isinstance(f, str):
            result.append(str_comp(f, x))
        elif callable(f):
            result.append(call(f, x))
        else:
            result.append(f == x)
    return result


# todo think?
# todo temporal backward compatibility
test_all = comp([test, all])
test_any = comp([test, any])


def filter_keys(d, keys, reverse=False, str_comp="=", *args, **kwargs):
    fn = test_any
    if dataclasses.is_dataclass(keys):
        keys = dataclasses.asdict(keys)
    if isinstance(keys, dict):
        keys = list(keys.keys())
    if dataclasses.is_dataclass(d):
        d = dataclasses.asdict(d)
    if reverse:
        fn = comp(test_any, lambda x: not x)
    return {k: v for k, v in d.items() if fn(keys, k, str_comp=str_comp, *args, **kwargs)}


def filter_init_args(base_class, params, **kwargs):
    return filter_keys({**params, **kwargs}, get_init_args(base_class))


# todo use print_arg
def filter_init(base_class, *args, **kwargs):
    kwargs = filter_init_args(base_class, params=kwargs)
    # logger.debug(f"{}")
    return base_class(*args, **kwargs)


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


# https://stackoverflow.com/questions/17215400/format-string-unused-named-arguments
def format_(str_, **kwargs):
    return str_.format_map(SafeDict(**kwargs))


def is_model(obj):
    return isinstance(obj, (Model, Layer, Loss))


def has_attr(obj, attr, empty=False):
    if empty:
        return hasattr(obj, attr) and getattr(obj, attr)
    return hasattr(obj, attr) and getattr(obj, attr) is not None


def lu(data):
    if hasattr(data, "__len__") and len(data) == 1:
        return data[0]
    return data


# copy from tensorflow2.4.1/engine/data_adapter
def is_array(v, df_include=False):
    """Return True if v is a Tensor, array, or is array-like."""
    if df_include:
        return (
                hasattr(v, "__getitem__") and
                hasattr(v, "shape") and
                (hasattr(v, "dtype") or hasattr(v, "dtypes")) and
                hasattr(v, "__len__")
        )
    return (
            hasattr(v, "__getitem__") and
            hasattr(v, "shape") and
            hasattr(v, "dtype") and
            hasattr(v, "__len__")
    )


def is_info(x):
    return il(x) and len(x) == 2 and isinstance(x[1], (str, type))


def ii(data):
    return isinstance(data, collections.abc.Iterable) and not isinstance(data, str)


def is_int(obj, bool_=False):
    if not bool_ and isinstance(obj, bool):
        return False
    return isinstance(obj, (int, np.integer))


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


def int_(obj, if_err=None):
    try:
        return int(obj)
    except:
        if if_err == "input":
            return obj
        return if_err


def get_input_layer(inputs=None, batch=None):
    # if isinstance(inputs, DataGenerator):
    #     inputs = inputs[0]
    # elif isinstance(inputs, DataAdapter):
    #     inputs= inputs[0:1]
    def check_batch(b):
        return is_int(b) or b is None

    if isinstance(inputs, dict):
        x = list(inputs.values())
        result = {}
    elif is_info(inputs):
        x = [inputs]
        result = []
    else:
        x = inputs
        result = []
    x = lw(x, none_empty=False)
    for i, s in enumerate(x):
        if is_info(s):
            dtype_ = tf.as_dtype(s[1])
            s = s[0]
        else:
            dtype_ = None

        if hasattr(s, "shape"):
            shape = s.shape
        elif ii(s):
            shape = s
        elif isinstance(s, int):
            shape = [s]
        else:
            shape = [64, 64, 3]
        shape = ([batch] if check_batch(batch) else []) + list(shape)
        inp = Input(shape[1:], batch_size=shape[0], dtype=dtype_)
        if isinstance(inputs, dict):
            result[list(inputs.keys())[i]] = inp
        else:
            result.append(inp)
    return result


def is_dict(obj):
    return isinstance(obj, dict)


def get_dtype(obj):
    return obj.dtype if hasattr(obj, "dtype") else obj.dtype_


get_dtype_str = comp([get_dtype, str])


def get_shape_core(x, length=None, batch=slice(1, None)):
    if batch is True:
        batch = slice(None)
    elif batch is False:
        batch = slice(1, None)
    if has_attr(x, "shape"):
        shape = x.shape[batch]
    elif hasattr(x, "__len__"):
        logger.info("get_shape: No shape in the object. Using len")
        shape = (len(x),)
    else:
        logger.info("get_shape: No shape and len in the object. Setting shape to 1.")
        shape = (1,)
    if length:
        return tuple(lw(length), ) + tuple(shape)
    return tuple(shape)


def is_info(x):
    return il(x) and len(x) == 2 and isinstance(x[1], (str, type))


def get_info(x, length=None, batch=slice(1, None)):
    shape = get_shape_core(x, length=length, batch=batch)
    if is_array(x):
        return shape, get_dtype_str(x)
    return shape


def lazy_wrapper(func, default_rf=False):
    def wrapper(*args, rf=default_rf, **kwargs):
        if rf:
            def inne_func():
                return func(*args, **kwargs)

            return inne_func
        else:
            return func(*args, **kwargs)

    return wrapper


def get_shape(x, length=None, batch=slice(1, None), base_fn=get_info):
    if base_fn == "shape":
        base_fn = get_shape_core
    elif base_fn == "info":
        base_fn = get_info
    elif base_fn == "type":
        base_fn = get_dtype

    # todo change name
    if has_attr(x, "get_shape"):
        return x.get_shape(length=length, batch=batch, base_fn=base_fn)
    elif il(x):
        return tuple(lu([get_shape(d, length=length, batch=batch, base_fn=base_fn) for d in x]))
    elif isinstance(x, dict):
        return {k: get_shape(v, length=length, batch=batch, base_fn=base_fn) for k, v in x.items()}
    else:
        return base_fn(x, length, batch)


def _remove_nones(d: Union[dict, list]):
    if isinstance(d, dict):
        result = {}
        for k, v in d.items():
            if v is None:
                logger.warning(f"Removing None {k} from output tensor.")
            else:
                result[k] = _remove_nones(v)

    elif isinstance(d, list):
        result = []
        for i, v in enumerate(d):
            if v is None:
                logger.warning(f"Removing None {i} from output tensor.")
            else:
                result.append(_remove_nones(v))
    else:
        return d
    return result


def build_functional_model(model, inputs=None, batch=True, base=Model, name=None):
    x = get_input_layer(inputs, batch)
    # outputs = lu(x)
    x = x if is_dict(x) else lu(x)
    outputs = x
    for m in lw(model):
        outputs = m(outputs)
    return base(inputs=x, outputs=_remove_nones(outputs), name=name)


def build_functional(model, inputs_=None, mode="class", *args, batch_=True, base_=Model, name=None,
                     **kwargs):
    model = build_functional_model(
        model=model if (is_model(model) and mode == "auto") or mode != "class" else model(*args, **kwargs),
        # model= model ,
        inputs=get_shape(inputs_),
        batch=batch_,
        base=base_,
        name=name
    )
    return model


def get_show_shape(show_shape):
    if callable(show_shape):
        shape_ = show_shape
    elif show_shape == "no_batch":
        def shape_(tensor):
            return tuple(tensor.shape[1:])
    else:
        def shape_(tensor):
            return tuple(tensor.shape)

    return shape_


def extract_shape(a, show_shape):
    if isinstance(a, dict):
        return {k: extract_shape(v, show_shape) for k, v in a.items() if v is not None}
    elif il(a):
        return [extract_shape(v, show_shape) for v in a if v is not None]
    else:
        return show_shape(a)


def extract_input_shape(*args, show_shape_, **kwargs):
    return extract_shape(args, show_shape_), extract_shape(kwargs, show_shape_)


LOG_MESSAGE_TEMPLATE = "{prefix}{name} {part}: {shape}"


def log_shape(show_shape=None, name=None, log_input=True, log_output=True, prefix_sing='\t',
              logger_=(print, append_global_flow), in_fn=None, out_fn=None):
    show_shape = get_show_shape(show_shape)
    if not show_shape:
        return lambda x: x

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if prefix_sing:
                prefix = prefix_sing * GLOBAL_STORAGE[MODEL_DEPTH]
            if log_input:
                if not isinstance(log_input, str):
                    log_input_ = LOG_MESSAGE_TEMPLATE
                log_input_ = format_(log_input_, prefix=prefix, name=name, part="in",
                                     shape=print_output(extract_input_shape(*args, show_shape_=show_shape, **kwargs)))
                for l in lw(logger_):
                    l(log_input_)
                for in_fn_ in lw(in_fn):
                    in_fn_(*args, **kwargs)
            GLOBAL_STORAGE[MODEL_DEPTH] += 1
            out = fn(*args, **kwargs)
            GLOBAL_STORAGE[MODEL_DEPTH] -= 1
            if log_output:
                if not isinstance(log_output, str):
                    log_output_ = LOG_MESSAGE_TEMPLATE
                log_output_ = format_(log_output_, prefix=prefix, name=name, part="out",
                                      shape=print_output(extract_shape(out, show_shape)))
                for l in lw(logger_):
                    l(log_output_)
                for out_fn_ in lw(in_fn):
                    out_fn_(*args, **kwargs)
            return out

        return wrapper

    return deco


# This kind of code could be written better. I just need to get it working quickly.

NO_OTUPUT = "__NO_OUTPUT__"


def clear_output(a):
    if isinstance(a, dict):
        a = {k: clear_output(v) for k, v in a.items() if clear_output(v) != NO_OTUPUT}
        if a:
            return a
        return NO_OTUPUT
    elif il(a):
        a = [clear_output(v) for v in a if clear_output(v) != NO_OTUPUT]
        if len(a) == 1:
            return a[0]
        elif a:
            return a
        return NO_OTUPUT
    elif a:
        return a
    else:
        return NO_OTUPUT


def print_output(a):
    if isinstance(a, dict):
        a = clear_output(a)
        if isinstance(a, dict) and len(a) == 1:
            a = a[list(a.keys())[0]]
    elif il(a):
        a = clear_output(a)
        if il(a) and len(a) == 1:
            a = a[0]
    if a != NO_OTUPUT:
        return a
    else:
        return ""


def interleave_layers(model, post=("LN", "LD"), post_no=-1, pre=None, pre_no=0):
    if post is True:
        post = ("LN", "LD")
    model = interleave(
        [[inter] * (pre_no if pre_no > 0 else len(model) + pre_no) for
         inter in lw(pre)]
        + [model] +
        [[inter] * (post_no if post_no > 0 else len(model) + post_no) for inter
         in
         lw(post)])
    return model


dense = partial(Dense, activation="relu")
conv = partial(Conv2D, activation="relu")


@lazy_wrapper
def ConvSequential(model, name="conv", last_act="same", if_empty=None, base_class=conv, post=None, post_no=-1,
                   pre=None, pre_no=0, **kwargs):
    model = interleave_layers(lw(model), post=post, post_no=post_no, pre=pre, pre_no=pre_no)
    layers = []
    for i, m in enumerate(lw(model)):
        if is_model(m):
            layers.append(m)
        elif isinstance(m, str):
            layers.append(REGULARIZATION[m]())
        else:
            if not isinstance(m, dict):
                params = dict_from_list(get_init_args(base_class), lw(m))
            else:
                params = m
            if last_act != "same" and i == len(lw(model)) - 1:
                params = {**kwargs, "activation": last_act, **params}
            else:
                params = {**kwargs, **params}

            layers.append(base_class(**params))
    if if_empty != "seq" and not layers:
        return if_empty
    return Sequential(layers, name=name)


DenseSequential = partial(ConvSequential, name=None, base_class=dense)


def build_layer(layer, **kwargs):
    if isinstance(layer, int):
        new_layer = dense(layer, **kwargs)
    elif isinstance(layer, tuple):
        if all([isinstance(e, int) for e in layer]):
            new_layer = DenseSequential(layer, **kwargs)
        else:
            new_layer = dense(*layer)
    elif is_dict(layer):
        new_layer = dense(**{**kwargs, **layer})

    elif is_model(layer):
        new_layer = layer
    elif layer is None:
        new_layer = Pass()
    elif callable(layer) and len(get_args(layer)) > 0:
        new_layer = Lambda(layer)
    else:
        new_layer = layer()
    return new_layer


def pass_(x):
    return x


Pass = lambda: Lambda(pass_, "pass")


class SubClassingModel(Model):
    def __init__(self, model, name=None):
        super().__init__(name=name)
        self.model = lw(model)

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    def __repr__(self):
        return f"{self.name}: {[getattr(m, 'name', '') for m in getattr(self, 'model', [])]}"


class SubClassing(SubClassingModel):
    def call(self, x):
        for layer in self.model:
            # Lambda get only one argument.
            if isinstance(layer, (Model, Lambda)):
                x = layer(x)
            else:
                if il(x):
                    x = layer(*x)
                else:
                    x = layer(x)
        return x


@lazy_wrapper
def SequentialModel(layers, name=None, add_flatten=True, none_identity=False, trainable=True, base_class=Sequential,
                    **kwargs):
    if isinstance(base_class, str):
        if base_class == "sub_model":
            base_class = SubClassingModel
        elif base_class == "sub":
            base_class = SubClassing
        else:
            base_class = Sequential
    new_layers = []
    for i, layer in enumerate(lw(layers, none_empty=False)):
        if layer is None and not none_identity: continue
        new_layer = build_layer(layer=layer, **kwargs)
        new_layers.append(new_layer)

    return base_class(new_layers, name=name) if len(new_layers) > 1 else new_layers[0]


def take_element(x, index=0):
    return x[index]


def take_from_batch(x, index=0):
    return x[:, index]


def Get(index=0, lambda_layer=True, if_batch=False):
    if if_batch:
        take_fn = partial(take_element, index=index)
    else:
        take_fn = partial(take_from_batch, index=index)
    if lambda_layer:
        return Lambda(take_fn, name="get")
    return take_fn


Last = partial(Get, index=-1)
First = partial(Get, index=0)

POOLING_CORE = {
    # FLATTEN
    "F": Flatten,
    FLAT: Flatten,
    FLATTEN: Flatten,
    # "F2": Flat2d,
    # "flat2d": Flat2d,
    "F2": Flatten2D,
    "flat2d": Flatten2D,
    # POOLING
    MP: MaxPooling2D,
    AP: AveragePooling1D,
    GM: GlobalMaxPooling2D,
    GA: GlobalAveragePooling2D,
    GM1: GlobalMaxPooling1D,
    GA1: GlobalAveragePooling1D,
    LAST: Last,
    FIRST: First,
}
POOLING = {**POOLING_CORE, **{k.lower(): v for k, v in POOLING_CORE.items()}}


# int - index of input shape
# str - size of dim
# tuple - product of slice
# list - product of index of input shape
# None = -1 calculate automatically
# todo high Add dict indexing.
# todo before this check if this could be done by Einstein notation.
class IndexReshape(Layer):
    def __init__(self, target_shape=(0, (1, -1), -1), mode="auto"):
        self.target_shape = target_shape
        self.mode = mode
        super().__init__()

    def build(self, input_shape):
        target_shape = []
        for o in self.target_shape:
            if isinstance(o, tuple):
                t = np.prod(input_shape[slice(*o)])
            elif isinstance(o, list):
                t = np.prod([input_shape[i] for i in o])
            elif isinstance(o, str):
                t = int(o)
            elif o is None:
                t = -1
            else:
                t = input_shape[o]
            if t is None:
                t = -1
            target_shape.append(t)
        if self.mode == "auto" and -1 not in target_shape:
            target_product = np.prod(target_shape)
            input_product = np.prod(input_shape)
            if input_product > target_product and input_product % target_product == 0:
                tmp = input_product / target_product
                rest = []
                for i in reversed(input_shape):
                    tmp /= i
                    rest.append(i)
                    if tmp == 1:
                        target_shape.extend(reversed(rest))
                        break

        logger.info(f"Build Reshaper: {input_shape} -> {target_shape}")
        self.model = Reshape(target_shape=target_shape[1:], input_shape=input_shape[1:])

    def call(self, inputs):
        return self.model(inputs)


def Pooling(pooling, *args, show_shape=False, **kwargs):
    if il(pooling):
        return IndexReshape(pooling)
    if isinstance(pooling, str):
        pooling = POOLING[pooling](*args, **kwargs)
    elif is_model(pooling):
        pass
    elif callable(pooling):
        pooling = pooling(*args, **kwargs)
    elif pooling is None:
        pooling = lambda: Lambda(identity)

    if show_shape:
        return log_shape(show_shape, "Pooling")(pooling)
    return pooling


def Extractor(
        model="ef",
        pooling="max",
        weights='imagenet',
        projection=None,
        include_preprocessing=True,
        show_shape: Union[bool, Callable] = False,
        **kwargs
):
    # todo tmp change
    if pooling == "max":
        pooling = "GM"
    elif pooling == "avg":
        pooling = "GA"

    # todo tmp change
    if model in ["vir26", "mib32", "vir26_32", "mib16", "vi", "vib32"]:
        pooling = None
    else:
        pooling = Pooling(pooling, show_shape=show_shape)

    weights = weights
    include_preprocessing = include_preprocessing

    @log_shape(show_shape, "Extractor")
    def apply(x):
        nonlocal include_preprocessing
        nonlocal model

        # if show_shape:
        #     print("Extractor input shape: ", show_shape(x))

        if include_preprocessing and include_preprocessing is not True:
            x = SequentialModel(
                lw(include_preprocessing),
                name="pre",
                add_flatten=False
            )(x)

            include_preprocessing = False

        if not callable(model):
            model = get_extractor(
                model=model,
                include_top=False,
                weights=weights,
                include_preprocessing=include_preprocessing,
                **{
                    "data": tuple(x.shape),
                    **kwargs
                }
            )
        x = log_shape(show_shape, "Encoder")(model)(x)

        if isinstance(projection, int):
            x = Conv2D(projection, 1)(x)

        if pooling:
            # x = log_shape(show_shape, "Extractor pooling")(short(pooling))(x)
            x = pooling(x)

        return x

    return apply


MODELS = PassDict({
    "mns": MobileNetV3Small,
    "vgg16": VGG16,
    "vgg": VGG19,
})
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, \
    EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L

MODELS["ef"] = EfficientNetV2B0
MODELS["ef1"] = EfficientNetV2B1
MODELS["ef2"] = EfficientNetV2B2
MODELS["ef3"] = EfficientNetV2B3
MODELS["efs"] = EfficientNetV2S
MODELS["efm"] = EfficientNetV2M
MODELS["efl"] = EfficientNetV2L
MODELS[None] = MODELS["ef"]


def get_extractor(
        data=(None, 224, 224, 3),
        batch=True,
        model="ef",
        include_top=False,
        weights='imagenet',
        include_preprocessing=True,
        **kwargs
):
    if not il(data):
        # if include_preprocessing == "auto":
        #     preprocess(data, tf=True,ef=True),
        #     include_preprocessing=False
        data = data.shape
    if batch:
        data = data[1:]
    return filter_init(
        MODELS[model],
        **dict(
            include_top=include_top,
            input_shape=data,
            weights=weights,
            include_preprocessing=include_preprocessing,
            **kwargs
        )
    )


class InitialWeight(Layer):
    def __init__(self, initializer="random_normal", name="initialweight", **kwargs):
        super().__init__(name=name, **kwargs)
        self.initializer = initializer

    def build(self, input_shape):
        self.w = self.add_weight(
            name="iw",
            shape=tuple(input_shape[1:]),
            initializer=self.initializer,
            # trainable=self.trainable
        )
        super().build(input_shape)

    def call(self, x):
        return self.w


def build_model(model, *args, default_model=Model, **kwargs):
    if not is_model(model):
        if not callable(model):
            model = default_model
        model = model(*args, **kwargs)
    return model


class InferencePass(Layer):
    def __init__(self, model, print_=False):
        super().__init__()
        self.model = model
        self.print_ = print_

    def call(self, *args, training=False):
        if training:
            return self.model(*args)
        else:
            return lu(args)


def Predictor(model=(1000, 1000), output_size=10, last_act=None, **kwargs):
    return DenseSequential(lw(model) + [output_size], last_act=last_act, **kwargs)


def rgb(multiples=(1, 1, 1, 3)):
    def fn(x):
        return tf.tile(x, multiples=multiples)

    return Lambda(fn)

build_subclassing2 = partial(SequentialModel, base_class=SubClassing)
class Base(Model):
    def __init__(self, model=None, name=None, debug_=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = build_subclassing2(model)

    def call(self, inputs):
        return K.map_(inputs, self.model)

    def __repr__(self):
        return f"{self.name}: {self.model}"
        # return f"{get_str_name(self)}: {get_str_name(self.model)}"
class BatchModel(Base):

    def call(self, inputs):
        return K.map_batch(inputs, self.model)
class Loss_(Model):
    def __init__(self, lw=1.0, reduction_fn=tf.reduce_mean):
        super().__init__()
        self.lw = lw
        self.reduction_fn = reduction_fn

    def call(self, inputs):
        loss = self.lw * self.reduction_fn(inputs)
        self.add_loss(loss)
        return loss

def add_loss(model, *args, name=None, loss_class=Loss_, filter_=None, **kwargs):
    loss_ = loss_class(*args, **kwargs)
    if filter_:
        loss_ = filter_(loss_)
    return SubClassing([model, loss_], name=name)
class FilterList(Model):
    def __init__(self, model, index=None):
        super().__init__()
        self.model = model
        self.index = index
        self.filter_ = Get(index, if_batch=True)

    def call(self, inputs):
        result = self.model(self.filter_(inputs))
        return replace_at_index2(inputs, self.index, result)
def predict(x):
    return tf.argmax(x, axis=-1)
def Predict():
    return Lambda(lambda x: predict(x))

def list_to_dict(x, name="output"):
    return {n: x[i] for i, n in enumerate(name)}


def dict_to_list(x, name="output"):
    return [x[n] for n in name]


# dict -> list
class DictModel(Base):
    # model - list model
    def __init__(self, model, in_=OUTPUT, out=OUTPUT, name=None, debug_=False, **kwargs):
        if name is None:
            name = f"{'_'.join(lw(in_))}__{'_'.join(lw(out))}"
            # name = f"({','.join(lw(inputs))})({','.join(lw(outputs))}"
        super().__init__(model=model, name=name, debug_=debug_, **kwargs)
        self.input_name = lw(in_)
        self.output_name = lw(out)

    def call(self, inputs):
        x = dict_to_list(inputs, name=self.input_name)
        result = self.model(lu(x))
        # todo create dict only from tuple, not list
        return {**inputs, **list_to_dict(lw(result), name=self.output_name)}
class Trainer(Model, KerasModelHubMixin):
    def __init__(
            self,
            model,
            loss=None,
            predict=PREDICT,
            output=None,
            add_loss=True,
            model_wrap=True,
            loss_wrap=True,
            predict_wrap=True,
            name=None,
            **kwargs
    ):
        super().__init__()
        self.model, self.loss_fn, self.predict_fn, self.output_model = build_trainer(
            model,
            loss=loss,
            predict=predict,
            output=output,
            add_loss=add_loss,
            model_wrap=model_wrap,
            loss_wrap=loss_wrap,
            predict_wrap=predict_wrap
        )
        super().__init__(name=name, **kwargs)

    def call(self, x):
        for model in lw(self.model):
            x = model(x)
        x = self.predict_fn(x)
        for loss in lw(self.loss_fn):
            x = loss(x)
        return self.output_model(x)

    def save(self, filepath="~/tmp/model", *args, only_inference=False, **kwargs):
        filepath = os.path.expanduser(filepath)
        # not working
        # built = False
        # if only_inference and is_model(self.loss_fn) and self.loss_fn.built:
        #     bu = self.loss_fn
        #     self.loss_fn = None
        if filepath.startswith("hf://"):
            super().push_to_hub(repo_id=filepath[5:], *args, **kwargs)
        else:
            super().save(filepath=filepath, *args, **kwargs)
        # if only_inference:
        #     self.loss_fn = loss

    def export(self, *args, only_inference=True, **kwargs):
        self.save(*args, only_inference=only_inference, **kwargs)

def model_output(output=PREDICT, return_list=False):
    if callable(output):
        output_fn = output
    elif output == "all":
        if return_list:
            def output_fn(x):
                return list(x.values())
        else:
            def output_fn(x):
                return x
    else:
        output = lw(output)
        if return_list:
            if len(output) == 1:
                def output_fn(x):
                    return list(filter_keys(x, output).values())[0]
            else:
                def output_fn(x):
                    return list(filter_keys(x, output).values())
        else:
            def output_fn(x):
                return filter_keys(x, output)

    return Lambda(output_fn, name="model_output")

def build_trainer(
        model,
        loss=None,
        predict=PREDICT,
        output=None,
        add_loss=True,
        model_wrap=True,
        loss_wrap=True,
        predict_wrap=True,
):
    def check(obj, class_ref, wrap):
        return isinstance(obj, class_ref) or obj is None or not wrap

    model = [m if check(m, DictModel, model_wrap) else DictModel(m, in_=INPUTS if i == 0 else OUTPUT,
                                                                 out=OUTPUT, name="model") for i, m in
             enumerate(lw(model))]
    if hasattr(model, "output_name"):
        model_output_name = model[-1].output_name
    else:
        model_output_name = OUTPUT
    model_output_name = lw(model_output_name)
    predict_input = model_output_name[0]
    if isinstance(predict, str):
        if predict == "loss" and hasattr(loss, "predict_fn"):
            predict = loss.predict_fn

        else:
            if predict == "loss":
                logger.warning(f"Predict is set to loss, but {loss} doesn't have predict_fn attribute.")
            predict_input = predict_input if predict == PREDICT else predict
            predict = Predict()
    predict = predict if check(predict, DictModel, predict_wrap) else DictModel(predict,
                                                                                in_=predict_input,
                                                                                out=PREDICT,
                                                                                name="predict")
    if add_loss:
        loss = [build_dict_loss(lo) for lo in lw(loss)]
    losses = []
    for lo in lw(loss):
        if check(lo, DictModel, loss_wrap):
            losses.append(lo)
        else:
            class_ref = PassDictModel if isinstance(lo, Model) else DictLayer
            # todo in case when output_name don't exist add OUTPUT
            losses.append(class_ref(lo, in_=[TARGET] + model_output_name, out=LOSS, name="loss"))
    if output is None:
        if predict is None or not hasattr(predict, "output_name"):
            output = model_output(model_output_name)
        else:
            output = model_output(predict.output_name)
    elif il(output) or isinstance(output, str):
        output = model_output(output)
    # return losses, model, output, predict
    return model, losses, predict, output

bm = SequentialModel

class TakeDict:
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.fn = take_dict if is_dict(self.data) else take

    def __getitem__(self, item):
        return self.fn(self.data, item)
    
REGULARIZATION_CORE = {
    BN: lambda *args, **kwargs: BatchNormalization(*args, **kwargs),
    LN: lambda epsilon=1e-6, *args, **kwargs: LayerNormalization(epsilon=epsilon, *args, **kwargs),  # epsilon=1e-6
    # IN: lambda: InstanceNormalization(),
}

REGULARIZATION = {**REGULARIZATION_CORE, **{k.lower(): v for k, v in POOLING_CORE.items()}}


def Regularization(regularization, *args, show_shape=False, **kwargs):
    if isinstance(regularization, str):
        if regularization in REGULARIZATION:
            regularization = REGULARIZATION[regularization](*args, **kwargs)
        else:
            regularization = Activation(regularization)
    elif is_model(regularization):
        pass
    elif callable(regularization):
        regularization = regularization(*args, **kwargs)
    elif regularization is None:
        regularization = lambda: Lambda(identity)

    if show_shape:
        return log_shape(show_shape, "Regularization")(regularization)
    return regularization

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from typing import Union, Tuple


def get_item(self, item: Union[str, Tuple[str], int]) -> Union[Layer, Model]:
    """
    This function retrieves an item from the current object, it can be a layer or a submodule.

    Parameters:
    - item (str, tuple, int): The item to retrieve. Can be a string with '/' separated values indicating the path to the item, a tuple of strings indicating the path to the item, or an integer indicating the index of the item.

    Returns:
    - The item found or None
    """
    if isinstance(item, str):
        item = item.split('/')
        if len(item) > 1:
            item = tuple(item)
        else:
            return self.get_layer(name=item[0])
    if isinstance(item, tuple):
        layer = self
        for i in item:
            layer = layer[i]
        return layer
    if isinstance(self, Model):
        return self.layers[item]
    return self.submodules[item]

def keras_get_item():
    Model.__getitem__ = get_item

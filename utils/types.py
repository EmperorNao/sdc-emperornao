from torch import Tensor, FloatTensor, BoolTensor, LongTensor  # noqa
from typing import Callable, Any, Dict, List, Optional, Tuple, Union  # noqa
from collections import OrderedDict  # noqa


def make_tuple(value: Union[object, Tuple[object, object]]):
    if isinstance(value, tuple):
        return value
    return value, value


def tuple_to_int(val):
    return tuple(map(int, val))

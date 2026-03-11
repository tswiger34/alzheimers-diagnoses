import copy
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.container import ModuleList


def get_clones(module: torch.nn.Module, N: int) -> ModuleList:
    return ModuleList(modules=[copy.deepcopy(x=module) for i in range(N)])


def get_activation_fn(activation: Literal["relu", "gelu"] | str) -> Callable[[Tensor], Tensor]:
    """Utility function for getting the activation function by name.

    Supported activation functions inlcude:
    - `relu`
    - `gelu`

    Args:
        activation (Literal["relu", "gelu"]): The name of the activation function

    Returns:
        Callable[[Tensor], Tensor]: The callable activation function
    """

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise TypeError("Expected `activation` arg to be one of 'relu' or 'gelu', but got {}".format(activation))


# a module_register machnism
# When a class @register("zero123-system") is applied, it stores zero123-system as the key in __modules__

__modules__ = {} # a global dictionary



# decorator, receive a function or class, and return a modified function or class
# here, "register" is a decorator, "decorator" is the inner func of decorator for wrapping
def register(name):
    def decorator(cls):
        __modules__[name] = cls
        return cls

    return decorator


# so i can find the "zero123-system" in systems/zero123/
def find(name):
    return __modules__[name]


###  grammar sugar for logging utilities  ###
import logging

logger = logging.getLogger("pytorch_lightning")

from pytorch_lightning.utilities.rank_zero import (
    rank_zero_debug,
    rank_zero_info,
    rank_zero_only,
)

debug = rank_zero_debug
info = rank_zero_info


@rank_zero_only
def warn(*args, **kwargs):
    logger.warn(*args, **kwargs)


from . import data, models, systems

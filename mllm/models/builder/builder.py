from typing import Dict, Any, Tuple

from torch import nn

from .build_perceptionGPT import load_pretrained_perceptionGPT

PREPROCESSOR = Dict[str, Any]


# TODO: Registry
def load_pretrained(model_args, training_args) -> Tuple[nn.Module, PREPROCESSOR]:
    type_ = model_args.type
    if 'shikra' in type_:
        return load_pretrained_perceptionGPT(model_args, training_args)
    else:
        assert False

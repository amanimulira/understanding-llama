import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig

from activations import ACT2FN
from cache_utils import Cache, DynamicCache
from generation import GenerationMixin
integrations.hub_kernels import use_kernel_forward_from_hub

"""
	`more imports`
"""

logger = logging.get_logger(__name__)



"""Model registry for EP08 INR super-resolution experiments."""

from ep08.models.deep_decoder import DeepDecoder
from ep08.models.siren import Siren, SirenLayer
from ep08.models.wire import Wire, WireLayer

__all__ = [
    "DeepDecoder",
    "Siren",
    "SirenLayer",
    "Wire",
    "WireLayer",
]

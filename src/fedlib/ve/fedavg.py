from . import base
from ..lib.algo import fedavg as trainer
from ..utils import get_logger
__all__ = ['fedavg']
class fedavg(base):
    base.trainer = trainer(get_logger())
    
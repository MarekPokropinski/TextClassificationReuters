from abc import abstractmethod
from typing import *


class BaseEmbedingModel:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.call(*args, **kwds)

    @abstractmethod
    def call(self, sequence):        
        raise NotImplementedError()
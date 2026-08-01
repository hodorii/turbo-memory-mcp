from typing import Dict, Type
from .interfaces import BaseQuantizer

class QuantizerRegistry:
    """
    양자화 알고리즘 구현체들을 관리하고 동적으로 로드하는 레지스트리 클래스.
    """
    _registry: Dict[str, Type[BaseQuantizer]] = {}

    @classmethod
    def register(cls, algo_id: str):
        """
        데코레이터를 통해 알고리즘 클래스를 레지스트리에 등록한다.
        """
        def wrapper(wrapped_class: Type[BaseQuantizer]):
            if not issubclass(wrapped_class, BaseQuantizer):
                raise TypeError(f"Class {wrapped_class.__name__} must inherit from BaseQuantizer")
            cls._registry[algo_id] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def get_quantizer(cls, algo_id: str, **kwargs) -> BaseQuantizer:
        """
        알고리즘 ID를 사용하여 구현체 인스턴스를 생성하여 반환한다.
        """
        if algo_id not in cls._registry:
            raise ValueError(f"Algorithm ID '{algo_id}' not found in registry. Available: {list(cls._registry.keys())}")
        
        quantizer_class = cls._registry[algo_id]
        return quantizer_class(**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """등록된 모든 알고리즘 ID 목록을 반환한다."""
        return list(cls._registry.keys())

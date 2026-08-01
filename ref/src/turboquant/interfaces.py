from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class QuantizedResult:
    """양자화된 결과 및 복원을 위한 메타데이터를 담는 데이터 클래스"""
    algo_id: str
    values: np.ndarray  # 양자화된 좌표 값
    signs: Optional[np.ndarray] = None  # 잔차의 부호 (회전 공간 기준)
    scale: Optional[float] = None      # 복원을 위한 최적 스케일 S

class BaseQuantizer(ABC):
    """
    모든 양자화 알고리즘이 구현해야 하는 표준 인터페이스.
    새로운 알고리즘 추가 시 이 클래스를 상속받아 구현한다.
    """
    
    @abstractmethod
    def quantize(self, x: np.ndarray) -> QuantizedResult:
        """
        입력 벡터 x를 양자화하여 QuantizedResult를 반환한다.
        
        Args:
            x (np.ndarray): 양자화할 입력 벡터 (d-dimensional)
            
        Returns:
            QuantizedResult: 양자화 결과 및 메타데이터
        """
        pass

    @abstractmethod
    def decode(self, q: QuantizedResult) -> np.ndarray:
        """
        양자화된 결과 q를 사용하여 원래 벡터 x를 복원한다.
        
        Args:
            q (QuantizedResult): 양자화된 결과 데이터
            
        Returns:
            np.ndarray: 복원된 벡터 x_hat
        """
        pass

    @abstractmethod
    def calculate_score(self, query: np.ndarray, q: QuantizedResult) -> float:
        """
        쿼리 벡터와 양자화된 결과 사이의 보정된 내적 점수를 계산한다.
        
        Args:
            query (np.ndarray): 검색 쿼리 벡터
            q (QuantizedResult): 양자화된 결과 데이터
            
        Returns:
            float: 보정된 내적 점수
        """
        pass

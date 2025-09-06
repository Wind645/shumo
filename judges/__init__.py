# Re-export all judge classes for backward compatibility
from .judge import OcclusionJudge, OcclusionResult
from .vectorized_judge import VectorizedOcclusionJudge, VectorizedOcclusionResult, vectorized_circle_fully_occluded_by_sphere
from .rough_judge import RoughOcclusionJudge, RoughVectorizedOcclusionJudge

try:
    from .vectorized_judge_torch import TorchVectorizedOcclusionJudge
    from .rough_judge_torch import TorchRoughVectorizedOcclusionJudge
    from .vectorized_judge_torch_sampled import TorchVectorizedOcclusionJudgeSampled
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    'OcclusionJudge', 'OcclusionResult',
    'VectorizedOcclusionJudge', 'VectorizedOcclusionResult', 
    'RoughOcclusionJudge', 'RoughVectorizedOcclusionJudge'
]

if TORCH_AVAILABLE:
    __all__.extend([
        'TorchVectorizedOcclusionJudge',
        'TorchRoughVectorizedOcclusionJudge', 
        'TorchVectorizedOcclusionJudgeSampled'
    ])
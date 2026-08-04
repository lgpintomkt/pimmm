"""
Physics-Informed Marketing Mix Modeling (PI-MMM) Package
--------------------------------------------------------
Implements Generalized Innovation-Diffusion (GID) and Generalized Bass Model (GBM)
state dynamics over machine learning and tabular foundation response surfaces.
"""

from .base import PhysicsInformedMMM
from .wrappers import TabFMResponseWrapper, MeridianResponseWrapper

__version__ = "1.0.0"
__all__ = [
    "PhysicsInformedMMM",
    "TabFMResponseWrapper",
    "MeridianResponseWrapper",
]
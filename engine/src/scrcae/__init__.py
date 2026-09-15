"""Supply Chain Resilience & Capital Allocation Engine.

A headless, testable extraction of the acquired Streamlit application's
optimization and stochastic simulation core.
"""

from .audit import ENGINE_VERSION, content_hash
from .calibration import (
    CALIBRATION_VERSION,
    DeliveryObservation,
    ElasticityFit,
    FitQuality,
    calibrate_elasticity,
)
from .domain import Bundle, Dependency, Intervention, Resource, SupplyNetwork
from .optimization import (
    MinimizeCapitalObjective,
    MonetaryNPVObjective,
    LegacyWeightedObjective,
    OptimizationRequest,
    PriceBook,
    solve,
)
from .risk import AllocationConcaveResponse, LinearResponse, ParameterPowerResponse
from .stochastic import (
    LognormalShock,
    LegacyTruncatedNormalShock,
    SimulationRequest,
    UniformCorrelation,
)

__version__ = ENGINE_VERSION

__all__ = [
    "AllocationConcaveResponse",
    "Bundle",
    "CALIBRATION_VERSION",
    "Dependency",
    "DeliveryObservation",
    "ENGINE_VERSION",
    "ElasticityFit",
    "FitQuality",
    "Intervention",
    "LegacyTruncatedNormalShock",
    "LegacyWeightedObjective",
    "LinearResponse",
    "LognormalShock",
    "MinimizeCapitalObjective",
    "MonetaryNPVObjective",
    "OptimizationRequest",
    "ParameterPowerResponse",
    "PriceBook",
    "Resource",
    "SimulationRequest",
    "SupplyNetwork",
    "UniformCorrelation",
    "calibrate_elasticity",
    "content_hash",
    "solve",
]

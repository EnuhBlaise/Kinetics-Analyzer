"""
Workflow module containing different kinetic model implementations.

Available workflows:
- SingleMonodWorkflow: Simple substrate limitation (no oxygen dynamics)
- SingleMonodLagWorkflow: Substrate limitation with lag phase (no oxygen dynamics)
- DualMonodWorkflow: Substrate + oxygen limitation with reaeration
- DualMonodLagWorkflow: Full model with lag phase
"""

from .base_workflow import BaseWorkflow, WorkflowResult
from .single_monod import SingleMonodWorkflow
from .single_monod_lag import SingleMonodLagWorkflow
from .dual_monod import DualMonodWorkflow
from .dual_monod_lag import DualMonodLagWorkflow

__all__ = [
    "BaseWorkflow",
    "WorkflowResult",
    "SingleMonodWorkflow",
    "SingleMonodLagWorkflow",
    "DualMonodWorkflow",
    "DualMonodLagWorkflow",
]

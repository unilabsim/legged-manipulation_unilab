from typing import Any

import numpy as np
from unilab.dtype_config import get_global_dtype
from unisim.dr.types import (
    INTERVAL_TERM_PUSH,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    IntervalTermOp,
)


def zero_actions(num_reset: int, num_action: int) -> np.ndarray:
    return np.zeros((num_reset, num_action), dtype=get_global_dtype())


def build_interval_push_plan(env: Any, step_counter: int) -> IntervalRandomizationPlan | None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None or not getattr(domain_rand, "push_robots", False):
        return None
    if step_counter % domain_rand.push_interval != 0:
        return None
    return IntervalRandomizationPlan(
        ops=(IntervalTermOp(INTERVAL_TERM_PUSH, np.asarray(domain_rand.max_force)),)
    )


def validate_interval_push_support(env: Any, capabilities: DomainRandomizationCapabilities) -> None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None or not getattr(domain_rand, "push_robots", False):
        return
    if not capabilities.supports_interval_term(INTERVAL_TERM_PUSH):
        raise NotImplementedError(
            f"{env._backend.backend_type} backend does not support interval push"
        )
    force_limit = np.asarray(domain_rand.max_force, dtype=np.float64)
    if force_limit.shape != (3,):
        raise ValueError(f"domain_rand.max_force must have shape (3,), got {force_limit.shape}")

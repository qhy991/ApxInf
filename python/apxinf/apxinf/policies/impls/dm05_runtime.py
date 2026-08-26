"""Explicit selector factory for DM05 inference runtimes."""

from __future__ import annotations

from typing import Any, Literal

DM05RuntimeSelector = Literal["default", "default_exact_combined"]
DEFAULT_DM05_RUNTIME_SELECTOR: DM05RuntimeSelector = "default"
EXACT_COMBINED_DM05_RUNTIME_SELECTOR: DM05RuntimeSelector = (
    "default_exact_combined"
)
DM05_RUNTIME_SELECTORS = frozenset(
    {DEFAULT_DM05_RUNTIME_SELECTOR, EXACT_COMBINED_DM05_RUNTIME_SELECTOR}
)

__all__ = [
    "DEFAULT_DM05_RUNTIME_SELECTOR",
    "DM05_RUNTIME_SELECTORS",
    "DM05RuntimeSelector",
    "EXACT_COMBINED_DM05_RUNTIME_SELECTOR",
    "create_dm05_runtime",
]


def create_dm05_runtime(
    model: Any,
    *,
    selector: DM05RuntimeSelector | str = DEFAULT_DM05_RUNTIME_SELECTOR,
    request_prefix_len: int = 564,
    diffusion_steps: int = 10,
) -> Any | None:
    """Create the explicit optimized runtime, or ``None`` for native default.

    Returning ``None`` preserves the existing model-owned default path.  The
    optimized import is lazy, so default users do not need Triton.  Unknown
    selectors raise rather than silently selecting another backend.
    """
    if selector == DEFAULT_DM05_RUNTIME_SELECTOR:
        return None
    if selector != EXACT_COMBINED_DM05_RUNTIME_SELECTOR:
        raise ValueError(
            f"Unsupported DM05 runtime selector {selector!r}; expected one of "
            f"{sorted(DM05_RUNTIME_SELECTORS)}."
        )
    from apxinf.policies.impls.dm05_combined_runtime import DM05CombinedRuntime

    return DM05CombinedRuntime(
        model,
        request_prefix_len=request_prefix_len,
        diffusion_steps=diffusion_steps,
    )

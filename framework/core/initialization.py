from __future__ import annotations

import torch


_INIT_TYPE_REGISTRY: dict[str, dict[str, str]] = {
    "near_zero_gaussian": {
        "label": "Near-zero Gaussian",
        "theory_note": "Gunasekar-style small random initialization near the origin.",
    },
}


def list_init_types() -> list[str]:
    return sorted(_INIT_TYPE_REGISTRY.keys())


def initialize_factor(
    *,
    seed: int,
    n: int,
    factor_rank: int,
    init_scale: float,
    init_type: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if init_type not in _INIT_TYPE_REGISTRY:
        supported = ", ".join(list_init_types())
        raise ValueError(f"Unsupported init_type={init_type!r}. Supported: {supported}.")

    torch.manual_seed(seed)

    if init_type == "near_zero_gaussian":
        return init_scale * torch.randn((n, factor_rank), device=device, dtype=dtype)

    raise ValueError(f"Unhandled init_type={init_type!r}.")

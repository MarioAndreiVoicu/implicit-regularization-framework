from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from framework.core.schedules import learning_rate_at_step


_OPTIMIZER_REGISTRY: dict[str, dict[str, str]] = {
    "gd": {
        "label": "Gradient Descent",
        "kind": "full_batch",
    },
    "adam": {
        "label": "Adam",
        "kind": "full_batch",
    },
    "sgd": {
        "label": "Minibatch SGD",
        "kind": "minibatch",
    },
    "momentum": {
        "label": "SGD + Momentum",
        "kind": "minibatch",
    },
    "coordinate_descent": {
        "label": "Coordinate Descent",
        "kind": "coordinate",
    },
    "preconditioned_gd": {
        "label": "Preconditioned GD",
        "kind": "full_batch",
    },
}


@dataclass
class BatchState:
    generator: torch.Generator
    permutation: torch.Tensor | None = None
    cursor: int = 0


@dataclass
class OptimizerState:
    tensors: dict[str, dict[str, torch.Tensor]]
    batch: BatchState | None = None
    coordinate_cursor: dict[str, int] | None = None
    step_count: int = 0


def list_optimizers() -> list[str]:
    return sorted(_OPTIMIZER_REGISTRY.keys())


def optimizer_kind(name: str) -> str:
    meta = _OPTIMIZER_REGISTRY.get(name)
    if meta is None:
        supported = ", ".join(list_optimizers())
        raise ValueError(f"Unsupported optimizer={name!r}. Supported: {supported}.")
    return meta["kind"]


def init_optimizer_state(*, optimizer: str, parameters: dict[str, torch.Tensor], seed: int) -> OptimizerState:
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in parameters.items():
        entry: dict[str, torch.Tensor] = {}
        if optimizer == "adam":
            entry["m"] = torch.zeros_like(tensor)
            entry["v"] = torch.zeros_like(tensor)
        elif optimizer == "momentum":
            entry["velocity"] = torch.zeros_like(tensor)
        tensors[name] = entry

    batch = None
    if optimizer_kind(optimizer) == "minibatch":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        batch = BatchState(generator=gen)

    coordinate_cursor = None
    if optimizer == "coordinate_descent":
        coordinate_cursor = {name: 0 for name in parameters.keys()}

    return OptimizerState(tensors=tensors, batch=batch, coordinate_cursor=coordinate_cursor, step_count=0)


def _next_minibatch_indices(*, m: int, batch_size: int, state: BatchState, device: torch.device) -> torch.Tensor:
    if state.permutation is None or state.cursor + batch_size > m:
        state.permutation = torch.randperm(m, generator=state.generator)
        state.cursor = 0
    sl = state.permutation[state.cursor : state.cursor + batch_size]
    state.cursor += batch_size
    return sl.to(device=device, dtype=torch.long)


def minibatch_indices(*, m: int, batch_size: int | None, state: OptimizerState, device: torch.device) -> torch.Tensor:
    if batch_size is None:
        raise ValueError("batch_size must be provided for minibatch optimizers")
    if state.batch is None:
        raise ValueError("minibatch_indices called without batch state")
    return _next_minibatch_indices(m=m, batch_size=batch_size, state=state.batch, device=device)


def zero_grad(parameters: dict[str, torch.Tensor]) -> None:
    for tensor in parameters.values():
        tensor.grad = None


def _coordinate_update_inplace(
    *,
    name: str,
    tensor: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
    cursor: int,
) -> int:
    flat = tensor.view(-1)
    flat_grad = grad.view(-1)
    idx = cursor % flat.numel()
    flat[idx] -= lr * flat_grad[idx]
    return idx + 1


def current_learning_rate(*, spec: Any, step_index: int) -> float:
    return learning_rate_at_step(
        base_lr=spec.lr,
        schedule_type=spec.lr_schedule_type,
        step=step_index,
        total_steps=spec.steps,
        decay_gamma=spec.lr_decay_gamma,
        step_size=spec.lr_step_size,
        lr_min=spec.lr_min,
    )


def apply_optimizer_step(
    *,
    optimizer: str,
    spec: Any,
    parameters: dict[str, torch.Tensor],
    state: OptimizerState,
) -> float:
    lr_t = current_learning_rate(spec=spec, step_index=state.step_count)
    adam_step = state.step_count + 1
    state.step_count += 1

    if optimizer in {"gd", "sgd"}:
        for tensor in parameters.values():
            if tensor.grad is None:
                raise ValueError("Missing gradient for optimizer step")
            tensor -= lr_t * tensor.grad
        return lr_t

    if optimizer == "momentum":
        for name, tensor in parameters.items():
            grad = tensor.grad
            if grad is None:
                raise ValueError("Missing gradient for optimizer step")
            velocity = state.tensors[name]["velocity"]
            velocity.mul_(spec.momentum_beta).add_(grad, alpha=-lr_t)
            tensor.add_(velocity)
        return lr_t

    if optimizer == "adam":
        for name, tensor in parameters.items():
            grad = tensor.grad
            if grad is None:
                raise ValueError("Missing gradient for optimizer step")
            slot = state.tensors[name]
            m = slot["m"]
            v = slot["v"]
            m.mul_(spec.adam_beta1).add_(grad, alpha=1.0 - spec.adam_beta1)
            v.mul_(spec.adam_beta2).addcmul_(grad, grad, value=1.0 - spec.adam_beta2)
            m_hat = m / (1.0 - spec.adam_beta1**adam_step)
            v_hat = v / (1.0 - spec.adam_beta2**adam_step)
            tensor.addcdiv_(m_hat, torch.sqrt(v_hat) + spec.adam_eps, value=-lr_t)
        return lr_t

    if optimizer == "coordinate_descent":
        if state.coordinate_cursor is None:
            raise ValueError("Coordinate-descent state is missing cursor")
        for name, tensor in parameters.items():
            grad = tensor.grad
            if grad is None:
                raise ValueError("Missing gradient for optimizer step")
            state.coordinate_cursor[name] = _coordinate_update_inplace(
                name=name,
                tensor=tensor,
                grad=grad,
                lr=lr_t,
                cursor=state.coordinate_cursor[name],
            )
        return lr_t

    if optimizer == "preconditioned_gd":
        if spec.parameterization == "factorized_psd":
            u = parameters["U"]
            grad = u.grad
            if grad is None:
                raise ValueError("Missing gradient for optimizer step")
            rank = u.shape[1]
            gram = u.transpose(0, 1) @ u
            eye = torch.eye(rank, device=u.device, dtype=u.dtype)
            precond = torch.linalg.inv(gram + spec.preconditioner_eps * eye)
            u -= lr_t * (grad @ precond)
            return lr_t
        if spec.parameterization == "factorized_uv":
            u = parameters["U"]
            v = parameters["V"]
            grad_u = u.grad
            grad_v = v.grad
            if grad_u is None or grad_v is None:
                raise ValueError("Missing gradient for optimizer step")
            rank = u.shape[1]
            eye_u = torch.eye(rank, device=u.device, dtype=u.dtype)
            eye_v = torch.eye(rank, device=v.device, dtype=v.dtype)
            gram_u = u.transpose(0, 1) @ u
            gram_v = v.transpose(0, 1) @ v
            precond_u = torch.linalg.inv(gram_v + spec.preconditioner_eps * eye_v)
            precond_v = torch.linalg.inv(gram_u + spec.preconditioner_eps * eye_u)
            u -= lr_t * (grad_u @ precond_u)
            v -= lr_t * (grad_v @ precond_v)
            return lr_t
        raise ValueError("preconditioned_gd requires factorized_psd or factorized_uv")

    supported = ", ".join(list_optimizers())
    raise ValueError(f"Unsupported optimizer={optimizer!r}. Supported: {supported}.")

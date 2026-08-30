"""Logical dtype conversion and raw tensor I/O shared by test tooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def float32_to_bfloat16_bits(value: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 (round-to-nearest-even) and return uint16 bits."""

    source = np.asarray(value, dtype=np.float32)
    bits = source.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = ((bits + bias) >> np.uint32(16)).astype(np.uint16)
    return np.where(np.isnan(source), np.uint16(0x7FC0), rounded).astype(np.uint16)


def bfloat16_bits_to_float32(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def encode_logical(value: np.ndarray, logical_dtype: str) -> np.ndarray:
    if logical_dtype == "float16":
        return np.asarray(value, dtype=np.float16)
    if logical_dtype == "bfloat16":
        return float32_to_bfloat16_bits(value)
    if logical_dtype == "float32":
        return np.asarray(value, dtype=np.float32)
    raise ValueError(f"unsupported logical dtype: {logical_dtype}")


def decode_logical(value: np.ndarray, logical_dtype: str) -> np.ndarray:
    if logical_dtype == "bfloat16":
        return bfloat16_bits_to_float32(value)
    return np.asarray(value, dtype=np.float32)


def storage_dtype(logical_dtype: str) -> np.dtype:
    result = {
        "float16": np.dtype("<f2"),
        "bfloat16": np.dtype("<u2"),
        "float32": np.dtype("<f4"),
    }.get(logical_dtype)
    if result is None:
        raise ValueError(f"unsupported logical dtype: {logical_dtype}")
    return result


def write_raw(path: Path, value: np.ndarray, logical_dtype: str) -> None:
    array = np.asarray(value)
    encoded = array if logical_dtype == "bfloat16" and array.dtype == np.uint16 else (
        encode_logical(array, logical_dtype)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(encoded).tofile(path)


def read_raw(path: Path, shape: tuple[int, ...], logical_dtype: str) -> np.ndarray:
    expected = int(np.prod(shape, dtype=np.int64))
    value = np.fromfile(path, dtype=storage_dtype(logical_dtype))
    if value.size != expected:
        raise ValueError(f"{path}: expected {expected} elements, found {value.size}")
    return value.reshape(shape)

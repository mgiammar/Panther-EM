"""Backend functions for image warping on different computational devices.

This module provides two sets of backend functions:
1. NumPy backend using scikit-image for CPU computation
2. PyTorch CUDA backend using cuCIM for GPU computation

NOTE: The warp backend functions only do interpolation and have no sense of quadrature
      of the coordinate transformation. Other code sources in Panther-EM handle scaling
      between polar/cartesian transformations.
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from skimage.transform import warp as skimage_warp

# Check for CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Optional imports for GPU support
try:
    import cupy as cp
    from cucim.skimage.transform import warp as cucim_warp

    CUCIM_AVAILABLE = True
except ImportError:
    CUCIM_AVAILABLE = False
    cp = None
    cucim_warp = None

# GPU support requires both CUDA and cuCIM
GPU_TRANSFORM_AVAILABLE = CUDA_AVAILABLE and CUCIM_AVAILABLE


# ============================================================================
# NumPy Backend (CPU)
# ============================================================================


def warp_numpy(
    image: np.ndarray,
    coords: np.ndarray,
    output_shape: tuple[int, int],
    order: int = 5,
    mode: str = "constant",
    cval: float = 0.0,
    clip: bool = False,
    **kwargs: dict[Any, Any],
) -> np.ndarray:
    """Warp an image using scikit-image (CPU).

    Parameters
    ----------
    image : np.ndarray
        Input image to warp.
    coords : np.ndarray
        Source coordinates for warping, shape (2, *output_shape).
    output_shape : tuple[int, int]
        Shape of the output image.
    order : int, optional
        Order of interpolation (0-5), by default 5.
    mode : str, optional
        How to handle values outside boundaries, by default "constant".
    cval : float, optional
        The constant value to use for padding when mode is "constant", by default 0.0.
    clip : bool, optional
        Whether to clip the output values to the range of the input image, by default
        False which is opposite of skimage.transform.warp's default behavior.
    **kwargs
        Additional arguments passed to skimage.transform.warp.

    Returns
    -------
    np.ndarray
        Warped image.
    """
    return skimage_warp(
        image,
        coords,
        output_shape=output_shape,
        order=order,
        mode=mode,
        cval=cval,
        clip=clip,
        **kwargs,
    )


# ============================================================================
# PyTorch CUDA Backend (GPU via cuCIM)
# ============================================================================


def warp_torch_cuda(
    image: torch.Tensor,
    coords: torch.Tensor,
    output_shape: tuple[int, int],
    order: int = 5,
    mode: str = "constant",
    cval: float = 0.0,
    clip: bool = False,
    **kwargs: dict[Any, Any],
) -> torch.Tensor:
    """Warp an image using cuCIM on GPU (for PyTorch CUDA tensors).

    This function converts PyTorch CUDA tensors to CuPy arrays, performs
    warping using cuCIM, and converts the result back to PyTorch tensors.

    Parameters
    ----------
    image : torch.Tensor
        Input image to warp (must be on CUDA device).
    coords : torch.Tensor
        Source coordinates for warping, shape (2, *output_shape).
        Must be on the same CUDA device as image.
    output_shape : tuple[int, int]
        Shape of the output image.
    order : int, optional
        Order of interpolation (0-5), by default 5.
    mode : str, optional
        How to handle values outside boundaries, by default "constant".
    cval : float, optional
        The constant value to use for padding when mode is "constant", by default 0.0.
    clip : bool, optional
        Whether to clip the output values to the range of the input image, by default
        False which is opposite of skimage.transform.warp's default behavior.
    **kwargs
        Additional arguments passed to cucim.skimage.transform.warp.

    Returns
    -------
    torch.Tensor
        Warped image on the same CUDA device as input.

    Raises
    ------
    ImportError
        If cuCIM/CuPy are not installed.
    RuntimeError
        If input tensors are not on CUDA device or CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            "CUDA is not available. Check your PyTorch installation and GPU setup."
        )

    if not CUCIM_AVAILABLE:
        raise ImportError(
            "CuPy and cuCIM are required for GPU warping. "
            "Install with: pip install cupy-cuda12x cucim"
        )

    # Validate inputs are on CUDA
    if not image.is_cuda:
        raise RuntimeError(
            f"Image must be on CUDA device for warp_torch_cuda, got {image.device}"
        )
    if not coords.is_cuda:
        raise RuntimeError(
            f"Coords must be on CUDA device for warp_torch_cuda, got {coords.device}"
        )
    if image.device != coords.device:
        raise RuntimeError(
            f"Image and coords must be on same device. "
            f"Got image: {image.device}, coords: {coords.device}"
        )

    device = image.device

    # Ensure current CUDA device matches tensor device for DLPack interop.
    # Use context manager to avoid changing global state permanently.
    with torch.cuda.device(device):
        # Convert PyTorch tensors to CuPy arrays (zero-copy via DLPack)
        image_cp = cp.from_dlpack(image.detach())
        coords_cp = cp.from_dlpack(coords.detach())

        # Perform warp on GPU using cuCIM
        warped_cp = cucim_warp(
            image_cp,
            coords_cp,
            output_shape=output_shape,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            **kwargs,
        )

    # Convert back to PyTorch tensor (zero-copy via DLPack)
    warped_torch = torch.from_dlpack(warped_cp)

    # Ensure output is on the correct device
    if warped_torch.device != device:
        warped_torch = warped_torch.to(device)

    return warped_torch


# ============================================================================
# Device Detection and Routing
# ============================================================================


def detect_device(array: np.ndarray | torch.Tensor) -> Literal["numpy"] | torch.device:
    """Detect whether an array is on CPU (numpy) or GPU (torch CUDA).

    Parameters
    ----------
    array : np.ndarray or torch.Tensor
        Input array.

    Returns
    -------
    str | torch.device
        "numpy" for CPU arrays or the concrete CUDA device for CUDA tensors.
    """
    if isinstance(array, torch.Tensor):
        if array.is_cuda:
            return array.device
        else:
            return "numpy"  # CPU torch tensors treated as numpy

    return "numpy"


def get_warp_function(device: Literal["numpy"] | torch.device) -> Callable:
    """Get the appropriate warp function for the specified device.

    Parameters
    ----------
    device : {"numpy"} or torch.device
        Device type. CUDA devices must be provided as torch.device.

    Returns
    -------
    callable
        Warp function for the specified device.

    Raises
    ------
    ValueError
        If device is not recognized.
    ImportError
        If required packages for the device are not installed.
    RuntimeError
        If CUDA is requested but not available.
    """
    if device == "numpy":
        return warp_numpy
    elif isinstance(device, torch.device) and device.type == "cuda":
        if not GPU_TRANSFORM_AVAILABLE:
            missing = []
            if not CUDA_AVAILABLE:
                missing.append("CUDA (check PyTorch installation and GPU)")
            if not CUCIM_AVAILABLE:
                missing.append("cuCIM (install with: uv sync --extra cuda12 | cuda13)")
            raise RuntimeError(
                f"GPU warping is not available. Missing: {', '.join(missing)}"
            )
        return warp_torch_cuda
    else:
        raise ValueError(
            f"Unsupported device: {device}. "
            "Supported: 'numpy' or torch.device('cuda[:index]')"
        )


def validate_same_device(
    *arrays: np.ndarray | torch.Tensor,
) -> Literal["numpy"] | torch.device:
    """Validate that all arrays are on the same device.

    Parameters
    ----------
    *arrays
        Arrays to validate.

    Returns
    -------
    str | torch.device
        The common device ("numpy" or a concrete CUDA torch.device).

    Raises
    ------
    ValueError
        If arrays are on different devices.
    RuntimeError
        If CUDA tensors are on different GPU devices.
    """
    if not arrays:
        return "numpy"

    devices = [detect_device(arr) for arr in arrays if arr is not None]

    if not devices:
        return "numpy"

    # Check all are same type (numpy vs cuda)
    first_device = devices[0]
    first_is_numpy = first_device == "numpy"
    if not all((d == "numpy") == first_is_numpy for d in devices):
        raise ValueError(
            f"All arrays must be on the same device type. Got: {set(devices)}"
        )

    # If CUDA, check all on same GPU device
    if first_device != "numpy":
        cuda_devices = [arr.device for arr in arrays if isinstance(arr, torch.Tensor)]
        if not all(d == cuda_devices[0] for d in cuda_devices):
            raise RuntimeError(
                f"All CUDA tensors must be on the same GPU device. "
                f"Got: {set(cuda_devices)}"
            )

    return first_device


def ensure_device(
    array: np.ndarray | torch.Tensor, target_device: Literal["numpy"] | torch.device
) -> np.ndarray | torch.Tensor:
    """Convert array to the target device if necessary.

    Parameters
    ----------
    array : np.ndarray or torch.Tensor
        Input array.
    target_device : {"numpy"} or torch.device
        Target device. CUDA targets must be provided as torch.device.

    Returns
    -------
    np.ndarray or torch.Tensor
        Array on the target device.

    Raises
    ------
    RuntimeError
        If CUDA is requested but not available.
    """
    current_device = detect_device(array)

    if current_device == target_device:
        return array

    if target_device == "numpy":
        # Convert to numpy
        if isinstance(array, torch.Tensor):
            return array.cpu().numpy()
        return np.asarray(array)

    elif isinstance(target_device, torch.device) and target_device.type == "cuda":
        # Convert to CUDA tensor
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA is not available. Check your PyTorch installation and GPU setup."
            )

        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).to(device=target_device)
        elif isinstance(array, torch.Tensor):
            return array.to(device=target_device)
        else:
            return torch.tensor(array, device=target_device)

    raise ValueError(f"Unsupported target device: {target_device}")

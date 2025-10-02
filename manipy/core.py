import functools
import inspect
import contextvars
from typing import Optional
import torch


_current_device_var = contextvars.ContextVar("manipy_device", default=None)
_current_dtype_var = contextvars.ContextVar("manipy_dtype", default=None)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Guard for environments without MPS backend
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def current_device() -> torch.device:
    """
    Return the current device set by the with_manipy decorator or a sensible default.
    """
    dev = _current_device_var.get()
    return dev if isinstance(dev, torch.device) else _default_device()


def current_dtype() -> torch.dtype:
    """
    Return the current dtype set by the with_manipy decorator or torch.float32.
    """
    dt = _current_dtype_var.get()
    return dt if isinstance(dt, torch.dtype) else torch.float32


def with_manipy(*, no_grad: bool = False, enable_grad: bool = False):
	"""
	Decorator that ensures torch-based functions have 'device' and 'dtype' kwargs.
	- device default: cuda > mps > cpu
	- dtype default: torch.float32
	- Autograd context controlled by decorator args: no_grad / enable_grad (mutually exclusive).
	"""
	if no_grad and enable_grad:
		raise ValueError("no_grad and enable_grad cannot both be True.")

	def decorator(fn):
		sig = inspect.signature(fn)
		has_varkw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
		accepts_device = "device" in sig.parameters or has_varkw
		accepts_dtype = "dtype" in sig.parameters or has_varkw

		@functools.wraps(fn)
		def wrapper(*args, **kwargs):
			# Resolve effective device
			effective_device: torch.device
			if "device" in kwargs and kwargs["device"] is not None:
				dev_val = kwargs["device"]
				if not isinstance(dev_val, torch.device):
					dev_val = torch.device(dev_val)
				effective_device = dev_val
			else:
				effective_device = _default_device()

			# Resolve effective dtype
			effective_dtype: torch.dtype
			if "dtype" in kwargs and kwargs["dtype"] is not None:
				dtype_val = kwargs["dtype"]
				if isinstance(dtype_val, str) and hasattr(torch, dtype_val):
					dtype_val = getattr(torch, dtype_val)
				effective_dtype = dtype_val
			else:
				effective_dtype = torch.float32

			# Set context vars so wrapped functions can access them via current_device/current_dtype
			_token_dev = _current_device_var.set(effective_device)
			_token_dtype = _current_dtype_var.set(effective_dtype)
			try:
				if accepts_device:
					kwargs["device"] = effective_device
				else:
					kwargs.pop("device", None)

				if accepts_dtype:
					kwargs["dtype"] = effective_dtype
				else:
					kwargs.pop("dtype", None)

				# Autograd context
				if no_grad:
					with torch.no_grad():
						return fn(*args, **kwargs)
				if enable_grad:
					with torch.enable_grad():
						return fn(*args, **kwargs)
				return fn(*args, **kwargs)
			finally:
				_current_device_var.reset(_token_dev)
				_current_dtype_var.reset(_token_dtype)

		return wrapper

	return decorator
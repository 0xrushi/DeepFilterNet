import os
import sys
from dataclasses import dataclass
from typing import Optional
import importlib.util


@dataclass(frozen=True)
class TorchCodecStatus:
    ok: bool
    error: Optional[BaseException] = None


_TORCHCODEC_CHECKED = False
_TORCHCODEC_STATUS: Optional[TorchCodecStatus] = None


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _preload_cuda_deps() -> None:
    """
    TorchCodec wheels may not carry an RPATH to CUDA runtime libraries shipped as
    pip packages under `site-packages/nvidia/**/lib`. On some setups this makes
    dlopen() of `libtorchcodec_core*.so` fail with missing CUDA libs (e.g.
    `libnppicc.so.12`, `libnvrtc.so.12`) even though torch itself works.

    Preloading these libraries with RTLD_GLOBAL allows the dynamic loader to
    satisfy TorchCodec's DT_NEEDED entries.
    """
    if not _is_linux():
        return

    try:
        import ctypes
        import glob

        import nvidia  # type: ignore
    except Exception:
        return

    # Keep this list short and targeted; expand only when we see additional
    # real-world missing dependencies.
    required = ("libnvrtc.so.12", "libnppicc.so.12")

    nvidia_roots = list(getattr(nvidia, "__path__", []))
    if not nvidia_roots:
        return

    for lib in required:
        lib_path = None
        for root in nvidia_roots:
            hits = glob.glob(os.path.join(root, "**", "lib", lib), recursive=True)
            if hits:
                lib_path = hits[0]
                break
        if lib_path is None:
            continue
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # If a preload fails, TorchCodec import will still report the root cause.
            pass


def check_torchcodec(require: bool = False) -> TorchCodecStatus:
    """
    Checks whether `torchcodec` can be imported and its native libraries can be loaded.

    If `require=True`, raises a RuntimeError with a focused hint for the most common
    missing dependency on CUDA wheels (`nvidia-npp-cu12`).
    """
    global _TORCHCODEC_CHECKED, _TORCHCODEC_STATUS

    if _TORCHCODEC_CHECKED and _TORCHCODEC_STATUS is not None:
        if require and not _TORCHCODEC_STATUS.ok:
            _raise_require_error(_TORCHCODEC_STATUS.error)
        return _TORCHCODEC_STATUS

    _preload_cuda_deps()
    try:
        import torchcodec  # noqa: F401

        _TORCHCODEC_STATUS = TorchCodecStatus(ok=True, error=None)
    except BaseException as e:
        # If torchcodec fails to import, probe the underlying dlopen error so we can provide
        # a precise hint (torchcodec's own error can hide the root missing dependency).
        probe = _probe_torchcodec_dlopen_error()
        if probe is not None:
            e = RuntimeError(f"{e}\n\n[torchcodec dlopen] {probe}")
        _TORCHCODEC_STATUS = TorchCodecStatus(ok=False, error=e)

    _TORCHCODEC_CHECKED = True

    if require and not _TORCHCODEC_STATUS.ok:
        _raise_require_error(_TORCHCODEC_STATUS.error)

    return _TORCHCODEC_STATUS


def _raise_require_error(error: Optional[BaseException]) -> None:
    msg = "TorchCodec is required but failed to load."
    detail = f"{error}" if error is not None else "unknown error"

    hint = ""
    if error is not None and "libnppicc.so.12" in str(error):
        hint = (
            "\n\nMissing CUDA NPP runtime library. Install `nvidia-npp-cu12` into the "
            "same environment as torch/torchaudio/torchcodec."
        )

    raise RuntimeError(msg + "\n\n" + detail + hint) from error


def _probe_torchcodec_dlopen_error() -> Optional[str]:
    if not _is_linux():
        return None

    spec = importlib.util.find_spec("torchcodec")
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg_dir = next(iter(spec.submodule_search_locations), None)
    if pkg_dir is None:
        return None

    try:
        import ctypes

        import torch  # noqa: F401
    except Exception:
        return None

    for v in (8, 7, 6, 5, 4):
        core = os.path.join(pkg_dir, f"libtorchcodec_core{v}.so")
        if not os.path.exists(core):
            continue
        try:
            ctypes.CDLL(core, mode=ctypes.RTLD_GLOBAL)
            return None
        except OSError as e:
            return str(e)

    return None

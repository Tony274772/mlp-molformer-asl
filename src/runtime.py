"""Runtime environment guards for constrained shared servers."""

import os


def configure_thread_limits(default_threads: str = "1") -> None:
    """Cap native thread pools before Torch/NumPy/RDKit/Transformers initialize."""
    thread_env_vars = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "RAYON_NUM_THREADS",
    )
    for name in thread_env_vars:
        os.environ.setdefault(name, default_threads)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRITON_DISABLE", "1")
    os.environ.setdefault("TORCH_USE_TRITON", "0")


def configure_torch_runtime(torch_module) -> None:
    """Apply Torch-specific thread and attention settings after importing torch."""
    try:
        torch_module.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if torch_module.backends.cuda.is_built():
        torch_module.backends.cuda.enable_flash_sdp(False)
        torch_module.backends.cuda.enable_mem_efficient_sdp(False)
        torch_module.backends.cuda.enable_math_sdp(True)

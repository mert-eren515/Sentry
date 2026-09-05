import os
from pathlib import Path


def enable():
    # onnxruntime-gpu needs the CUDA runtime DLLs on the DLL search path. NVIDIA
    # publishes those as nvidia-*-cu13 pip packages, but only for Linux - on
    # Windows they have no wheel and fail to build. The CUDA build of torch
    # ships the very same DLLs in torch/lib, and ultralytics pulls torch in for
    # the pose model anyway, so we borrow them instead of asking the user to
    # install the CUDA toolkit system-wide.
    try:
        import torch
        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
    except ImportError:
        pass

    import onnxruntime as ort
    ort.preload_dlls()

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        print("WARNING: CUDA not available, falling back to CPU")
    return providers

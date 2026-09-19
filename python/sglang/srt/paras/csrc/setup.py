from setuptools import setup
import sys

# Delay torch import to avoid issues with build isolation
def get_extensions():
    import os
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    unroll = os.environ.get("V3_UNROLL")
    nvcc_args = [
        "-O3",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
        "--expt-relaxed-constexpr",
    ]
    if unroll:
        nvcc_args.append(f"-DV3_UNROLL={int(unroll)}")
    return [
        CUDAExtension(
            name="paras_peer_access_cuda",
            sources=[
                "peer_access_transfer.cu",
                "kernels_v3.cu",
                "kernels_v3_cache.cu",
                "binding.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_args,
            },
        )
    ]

def get_cmdclass():
    from torch.utils.cpp_extension import BuildExtension
    return {"build_ext": BuildExtension}

setup(
    name="paras_peer_access_cuda",
    ext_modules=get_extensions(),
    cmdclass=get_cmdclass(),
)

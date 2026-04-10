from setuptools import setup
import sys

# Delay torch import to avoid issues with build isolation
def get_extensions():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    return [
        CUDAExtension(
            name="paras_peer_access_cuda",
            sources=["peer_access_transfer.cu", "binding.cpp"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-gencode=arch=compute_80,code=sm_80", "-gencode=arch=compute_90,code=sm_90", "--expt-relaxed-constexpr"],
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

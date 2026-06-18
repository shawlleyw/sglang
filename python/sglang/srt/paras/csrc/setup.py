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
    for env_name in ("WARPS_PER_BLOCK", "V3_MIN_BLOCKS_PER_SM"):
        v = os.environ.get(env_name)
        if v:
            nvcc_args.append(f"-D{env_name}={int(v)}")
    extra = os.environ.get("DPTP_NVCC_EXTRA")
    if extra:
        nvcc_args.extend(extra.split())
    return [
        CUDAExtension(
            name="paras_peer_access_cuda",
            sources=[
                "peer_access_transfer.cu",
                "kernels_v3.cu",
                "kernels_v3_cache.cu",
                "kernels_dptp.cu",
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

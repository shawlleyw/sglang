#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

// Forward declarations from .cu
void launch_peer_access_stub(const char* src, char* dst, size_t size);

PYBIND11_MODULE(paras_peer_access_cuda, m) {
    m.doc() = "ParaS CUDA peer access transfer kernels";
    m.def("stub_hello", []() { return std::string("paras_peer_access_cuda loaded"); });
    // Real functions will be added in Task 4
}

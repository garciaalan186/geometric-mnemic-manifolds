#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "kernel.cpp"

namespace py = pybind11;

// Wrapper function to handle NumPy input
std::vector<std::pair<float, int>> fast_scan_wrapper(
    py::array_t<float> data_array,
    py::array_t<int> indices_array,
    py::array_t<float> query_array,
    int k
) {
    auto data_buf = data_array.request();
    auto indices_buf = indices_array.request();
    auto query_buf = query_array.request();

    if (data_buf.ndim != 2) throw std::runtime_error("Data must be 2D");
    if (query_buf.ndim != 1) throw std::runtime_error("Query must be 1D");
    if (indices_buf.ndim != 1) throw std::runtime_error("Indices must be 1D");

    int n = data_buf.shape[0];
    int d = data_buf.shape[1];

    if (query_buf.shape[0] != d) throw std::runtime_error("Query dim mismatch");
    if (indices_buf.shape[0] != n) throw std::runtime_error("Indices length mismatch");

    float* data_ptr = static_cast<float*>(data_buf.ptr);
    int* indices_ptr = static_cast<int*>(indices_buf.ptr);
    float* query_ptr = static_cast<float*>(query_buf.ptr);

    return fast_scan(data_ptr, n, d, indices_ptr, query_ptr, k);
}

PYBIND11_MODULE(gmm_native, m) {
    m.doc() = "GMM Native C++ Kernel via PyBind11";
    m.def("fast_scan", &fast_scan_wrapper, "Fast Linear Scan with Dot Product");
}

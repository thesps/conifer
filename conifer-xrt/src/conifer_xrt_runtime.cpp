#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_ip.h"

class ConiferXilinxHLSKernelInfo {
  public:
  int n_features;
  int n_classes;

  ConiferXilinxHLSKernelInfo(int n_features_, int n_classes_)
    : n_features(n_features_), n_classes(n_classes_) {}

  ConiferXilinxHLSKernelInfo() : n_features(0), n_classes(0) {}
};

ConiferXilinxHLSKernelInfo get_conifer_xilinxhls_kernel_info(const xrt::device& device, const xrt::uuid& xclbin_id, const std::string& kernel_name) {
  // Read n_features and n_classes from kernel registers
  auto ip = xrt::ip(device, xclbin_id, kernel_name);
  int n_features = ip.read_register(0x18);
  int n_classes = ip.read_register(0x28);
  return ConiferXilinxHLSKernelInfo(n_features, n_classes);
}

class ConiferXilinxHLSXRTRuntime {
  public:
  xrt::device device;
  xrt::uuid xclbin_id;
  xrt::kernel kernel;
  ConiferXilinxHLSKernelInfo kernel_info;

  ConiferXilinxHLSXRTRuntime(const int device_index, const std::string& xclbin_path, const std::string& kernel_name_){
    device = xrt::device(device_index);
    xclbin_id = device.load_xclbin(xclbin_path);
    kernel_info = get_conifer_xilinxhls_kernel_info(device, xclbin_id, kernel_name_);
    kernel = xrt::kernel(device, xclbin_id, kernel_name_);
  }

  void allocate_buffers(size_t batch_size){
    // TODO allow for different data types
    size_t input_size = batch_size * kernel_info.n_features * sizeof(float);
    size_t output_size = batch_size * kernel_info.n_classes * sizeof(float);
    Xbo = allocate_bo(input_size);
    Ybo = allocate_bo(output_size);
  }

  pybind11::array_t<float> decision_function(pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> X){
    
    size_t batch_size;
    if(!_infer_batch_size(X, batch_size)){
      throw std::runtime_error("Input data size is not compatible with the number of features.");
    }
    if(_check_X_buffer_size(X) == false){
      throw std::runtime_error("Input buffer size does not match allocated buffer size.");
    }
    if(_check_Y_buffer_size(X) == false){
      throw std::runtime_error("Output buffer size does not match allocated buffer size.");
    }
    // Copy input data to device buffer
    pybind11::buffer_info X_info = X.request();
    //std::memcpy(Xbo.map(), X.data(), Xbo.size());
    Xbo.write(X_info.ptr);
    Xbo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Launch kernel
    auto run = kernel(batch_size, 0, 0, Xbo, Ybo);
    run.wait();

    // Copy output data back to host
    Ybo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Create numpy array from output buffer
    size_t n = batch_size * kernel_info.n_classes;
    auto Y = pybind11::array_t<float>(n);
    pybind11::buffer_info Y_info = Y.request();
    Ybo.read(Y_info.ptr);

    return Y;
  }

  private:

  xrt::bo Xbo;
  xrt::bo Ybo;

  xrt::bo allocate_bo(size_t size){
    return xrt::bo(device, size, kernel.group_id(3));
  }

  bool _infer_batch_size(const pybind11::array_t<float>& X, size_t& batch_size){
    if(X.size() % kernel_info.n_features != 0){
      return false;
    }
    batch_size = X.size() / kernel_info.n_features;
    return true;
  }

  bool _check_X_shape(const pybind11::array_t<float>& X, size_t batch_size){
    // check the shape of X against expected shape
    return X.size() == batch_size * kernel_info.n_features;
  }

  bool _check_Y_buffer_size(const pybind11::array_t<float>& X){
    // check the size of Y against the allocated buffer size
    size_t expected_size = Ybo.size() / sizeof(float);
    size_t actual_size = X.size() / kernel_info.n_features * kernel_info.n_classes;
    return expected_size == actual_size;
  }

  bool _check_X_buffer_size(const pybind11::array_t<float>& X){
    // check the size of X against the allocated buffer size
    size_t expected_size = Xbo.size() / sizeof(float);
    size_t actual_size = X.size();
    return expected_size == actual_size;
  }
};

PYBIND11_MODULE(conifer_xrt_runtime, m){
  pybind11::class_<ConiferXilinxHLSKernelInfo>(m, "ConiferXilinxHLSKernelInfo")
    .def(pybind11::init<int, int>(), pybind11::arg("n_features"), pybind11::arg("n_classes"))
    .def_readonly("n_features", &ConiferXilinxHLSKernelInfo::n_features)
    .def_readonly("n_classes", &ConiferXilinxHLSKernelInfo::n_classes);

  pybind11::class_<ConiferXilinxHLSXRTRuntime>(m, "ConiferXilinxHLSXRTRuntime")
    .def(pybind11::init<const int, const std::string&, const std::string&>(), pybind11::arg("device_index"), pybind11::arg("xclbin_path"), pybind11::arg("kernel_name"))
    .def("allocate_buffers", &ConiferXilinxHLSXRTRuntime::allocate_buffers)
    .def("decision_function", &ConiferXilinxHLSXRTRuntime::decision_function)
    /*.def_readonly("device", &ConiferXilinxHLSXRTRuntime::device)
    .def_readonly("xclbin_id", &ConiferXilinxHLSXRTRuntime::xclbin_id)
    .def_readonly("kernel", &ConiferXilinxHLSXRTRuntime::kernel)*/
    .def_readonly("kernel_info", &ConiferXilinxHLSXRTRuntime::kernel_info);
}


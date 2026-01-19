#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_ip.h"

#include "nlohmann/json.hpp"

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

class ConiferFPUKernelInfo {
  public:
  int nodes;
  int tree_engines;
  int features;
  bool dynamic_scaler;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(ConiferFPUKernelInfo, nodes, tree_engines, features, dynamic_scaler);

  ConiferFPUKernelInfo(std::string info_string){
    nlohmann::json j = nlohmann::json::parse(info_string);
    from_json(j.at("configuration"), *this);
  }

  // Default constructor
  ConiferFPUKernelInfo() : nodes(0), tree_engines(0), features(0), dynamic_scaler(false) {}
};

int get_conifer_fpu_kernel_info_length(const int device_index, const std::string& xclbin_path, const std::string& kernel_name) {
  xrt::device device = xrt::device(device_index);
  xrt::uuid xclbin_id = device.load_xclbin(xclbin_path);
  auto ip = xrt::ip(device, xclbin_id, kernel_name);
  ip.read_register(0x00); // dummy read to clear ap_start, ap_done
  ip.write_register(0x00, 1); // set ap_start to 1 to trigger info length read
  return ip.read_register(0x7C);
}

ConiferFPUKernelInfo get_conifer_fpu_kernel_info(const int device_index, const std::string& xclbin_path, const std::string& kernel_name) {

  // read the info length register to allocate the correct buffer size for the info
  int info_length = get_conifer_fpu_kernel_info_length(device_index, xclbin_path, kernel_name);
  if(info_length <= 0){
    throw std::runtime_error("Invalid info length retrieved from kernel: " + std::to_string(info_length));
  }

  // now load the kernel
  xrt::device device = xrt::device(device_index);
  xrt::uuid xclbin_id = device.load_xclbin(xclbin_path);
  auto kernel = xrt::kernel(device, xclbin_id, kernel_name, xrt::kernel::cu_access_mode::exclusive);

  // allocate buffer for info and dummy buffer for other args
  xrt::bo info_bo = xrt::bo(device, info_length * sizeof(char), kernel.group_id(9));
  xrt::bo dummy_bo = xrt::bo(device, 4, kernel.group_id(0));

  // retrieve the info from the kernel
  auto run = kernel(dummy_bo, dummy_bo, 0, 0, 0, dummy_bo, dummy_bo, dummy_bo, dummy_bo, info_bo, 0);
  run.wait();
  info_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::string info_str(static_cast<char*>(info_bo.map()), info_length);
  return ConiferFPUKernelInfo(info_str);
}

class ConiferFPUXRTRuntime {
  public:
  xrt::device device;
  xrt::uuid xclbin_id;
  xrt::kernel kernel;
  ConiferFPUKernelInfo fpu_info;

  xrt::bo Xbo;
  xrt::bo Ybo;

  ConiferFPUXRTRuntime(const int device_index, const std::string& xclbin_path, const std::string& kernel_name){
    device = xrt::device(device_index);
    xclbin_id = device.load_xclbin(xclbin_path);
    fpu_info = get_conifer_fpu_kernel_info(device_index, xclbin_path, kernel_name);
    kernel = xrt::kernel(device, xclbin_id, kernel_name);
  }

  void load(const pybind11::array_t<int>& nodes,
            const pybind11::array_t<float>& scales,
            const int batch_size,
            const int n_features,
            const int n_classes)
            {
    xrt::bo nodes_bo = xrt::bo(device, fpu_info.tree_engines * fpu_info.nodes * 7 * sizeof(int), kernel.group_id(5));
    xrt::bo scales_bo = xrt::bo(device, (fpu_info.features + 1) * sizeof(float), kernel.group_id(7));
    xrt::bo dummy_bo = xrt::bo(device, 4, kernel.group_id(0));

    // copy nodes data to device buffer
    pybind11::buffer_info nodes_info = nodes.request();
    nodes_bo.write(nodes_info.ptr);
    nodes_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // copy scales data to device buffer
    pybind11::buffer_info scales_info = scales.request();
    scales_bo.write(scales_info.ptr); 
    scales_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Launch kernel in load mode
    auto run = kernel(dummy_bo, dummy_bo, 1, 0, 0, nodes_bo, dummy_bo, scales_bo, dummy_bo, dummy_bo, 0);
    run.wait();

    allocate_buffers(batch_size, n_features, n_classes);
  }

  pybind11::array_t<int> read(){
    xrt::bo nodes_bo = xrt::bo(device, fpu_info.tree_engines * fpu_info.nodes * 7 * sizeof(int), kernel.group_id(5));
    xrt::bo scales_bo = xrt::bo(device, (fpu_info.features + 1) * sizeof(float), kernel.group_id(7));
    xrt::bo dummy_bo = xrt::bo(device, 4, kernel.group_id(0));

    // Launch kernel in read mode
    auto run = kernel(dummy_bo, dummy_bo, 2, 0, 0, dummy_bo, nodes_bo, dummy_bo, scales_bo, dummy_bo, 0);
    run.wait();
    nodes_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // Create numpy array from output buffer
    size_t n = fpu_info.tree_engines * fpu_info.nodes * 7;
    auto nodes = pybind11::array_t<int>(n);
    pybind11::buffer_info nodes_info = nodes.request();
    nodes_bo.read(nodes_info.ptr);
    return nodes;
  }

  void allocate_buffers(const size_t batch_size, const int n_features, const int n_classes){
    // TODO allow for different data types
    size_t input_size = batch_size * n_features * sizeof(float);
    size_t output_size = batch_size * n_classes * sizeof(float);
    Xbo = xrt::bo(device, input_size, kernel.group_id(0));
    Ybo = xrt::bo(device, output_size, kernel.group_id(1));
  }

  pybind11::array_t<float> decision_function(pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> X, size_t batch_size, size_t n_features, size_t n_classes){
  
    // Copy input data to device buffer
    pybind11::buffer_info X_info = X.request();
    //std::memcpy(Xbo.map(), X.data(), Xbo.size());
    Xbo.write(X_info.ptr);
    Xbo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Launch kernel
    xrt::bo dummy_bo = xrt::bo(device, 4, kernel.group_id(0));
    auto run = kernel(Xbo, Ybo, 3, batch_size, n_features, dummy_bo, dummy_bo, dummy_bo, dummy_bo, dummy_bo, 0);
    run.wait();

    // Copy output data back to host
    Ybo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Create numpy array from output buffer
    size_t n = batch_size * n_classes;
    auto Y = pybind11::array_t<float>(n);
    pybind11::buffer_info Y_info = Y.request();
    Ybo.read(Y_info.ptr);

    return Y;
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

  pybind11::class_<ConiferFPUKernelInfo>(m, "ConiferFPUKernelInfo")
    .def(pybind11::init<std::string>(), pybind11::arg("info_str"))
    .def_readonly("nodes", &ConiferFPUKernelInfo::nodes)
    .def_readonly("tree_engines", &ConiferFPUKernelInfo::tree_engines)
    .def_readonly("features", &ConiferFPUKernelInfo::features)
    .def_readonly("dynamic_scaler", &ConiferFPUKernelInfo::dynamic_scaler);

  pybind11::class_<ConiferFPUXRTRuntime>(m, "ConiferFPUXRTRuntime")
    .def(pybind11::init<const int, const std::string&, const std::string&>(), pybind11::arg("device_index"), pybind11::arg("xclbin_path"), pybind11::arg("kernel_name"))
    .def("load", &ConiferFPUXRTRuntime::load)
    .def("read", &ConiferFPUXRTRuntime::read)
    .def("decision_function", &ConiferFPUXRTRuntime::decision_function)
    .def_readonly("fpu_info", &ConiferFPUXRTRuntime::fpu_info);

    m.def("get_conifer_xilinxhls_kernel_info", &get_conifer_xilinxhls_kernel_info);
    m.def("get_conifer_fpu_kernel_info", &get_conifer_fpu_kernel_info);
    m.def("get_conifer_fpu_kernel_info_length", &get_conifer_fpu_kernel_info_length);
}


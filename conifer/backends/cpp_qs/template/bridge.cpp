#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "conifer_qs.h"
// conifer insert include

// conifer insert typedef

namespace py = pybind11;
PYBIND11_MODULE(conifer_bridge, m) { //create module => 'import conifer_bridge'
  py::class_<conifer_qs::BDT<BDTConfig>>(m, "BDT", py::module_local()) //expose conifer_bridge.BDT
      .def(py::init<const std::string &, int, int>(), py::arg("filename"),
           py::arg("tau") = 0, py::arg("delta") = 0) //expose constructor => conifer_bridge.BDT(filename, tau, delta)
      .def("nbytes", &conifer_qs::BDT<BDTConfig>::nbytes) //expose method
      .def(
          "decision_function",
          [](const conifer_qs::BDT<BDTConfig> &self,
             py::array_t<double, py::array::c_style | py::array::forcecast> X) { // C style: contiguous ROW MAJOR layout + cast to double
            auto buf = X.request();
            if (buf.ndim != 2) { //method expects (n_samples, n_features)
              throw std::runtime_error("Expected 2D input");
            }
            size_t n_samples = (size_t)buf.shape[0];
            std::vector<double> y = self.decision_function_batch(
                static_cast<const double *>(buf.ptr), n_samples,
                (size_t)buf.shape[1]);//access np array memory from c++ using pointer
            py::array_t<double> out({n_samples, (size_t)self.get_n_classes()});//np output array
            std::copy(y.begin(), y.end(), out.mutable_data());
            return out; //so that scores = bdt.decision_function(X) works...
          }); //expose method BUT use lambda wrapper to coonvert np array to c++ pointer
}

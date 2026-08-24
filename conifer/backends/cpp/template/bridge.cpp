#include "conifer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
// conifer insert include

// conifer insert typedef

namespace py = pybind11;
PYBIND11_MODULE(conifer_bridge, m){
  py::class_<conifer::BDT<BDTConfig>>(m, "BDT", py::module_local())
      .def(py::init<const std::string &>())
      .def("decision_function", &conifer::BDT<BDTConfig>::_decision_function_double)
      // set tau/delta as runtime arguments, so the block shape can be changed without recompiling the project
      .def("init_quickscorer", &conifer::BDT<BDTConfig>::init_quickscorer,
           py::arg("tau") = 0, py::arg("delta") = 0)
      .def("nbytes", &conifer::BDT<BDTConfig>::nbytes)
      .def("decision_function_batch",
          [](const conifer::BDT<BDTConfig> &self,
             py::array_t<double, py::array::c_style | py::array::forcecast> X){ // C style: contiguous row major layout + cast to double
            auto buf = X.request();
            if(buf.ndim != 2){ // method expects (n_samples, n_features)
              throw std::runtime_error("Expected 2D input");
            }
            size_t n_samples = (size_t) buf.shape[0];
            std::vector<double> y = self._decision_function_batch_double(
                static_cast<const double *>(buf.ptr), n_samples,
                (size_t) buf.shape[1]); // access the numpy array memory from c++ through a pointer
            py::array_t<double> out({n_samples, (size_t) self.get_n_classes()});
            std::copy(y.begin(), y.end(), out.mutable_data());
            return out;
          });
}

#include "conifer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// conifer insert include

// conifer insert typedef

namespace py = pybind11;
PYBIND11_MODULE(conifer_bridge, m){
  py::class_<conifer::BDT<T,U,W,false>>(m, "BDT", py::module_local())
      .def(py::init<const std::string &>())
      .def("decision_function", &conifer::BDT<T,U,W>::_decision_function_double);
}
#include "ap_fixed.h"
#include <pybind11/pybind11.h>

// conifer insert typedef

int to_int(double x){
  T y = (T) x;
  return y.V;
}

double to_double(double x){
  T y = (T) x;
  return y.to_double();
}

namespace py = pybind11;
PYBIND11_MODULE(fixed_point, m) {
    m.doc() = "fixed point conversion";
    m.def("to_int", &to_int, "Get the integer representation of the ap_fixed");
    m.def("to_double", &to_double, "Get the double representation of the ap_fixed");
}
#pragma once
#include <string>

#include "bdt_qs_obl.hpp"
#include "params.h"

using namespace adf;

namespace bdtmt {

constexpr unsigned N_IN = SPLIT_TREE ? 1u : N_TILES;
constexpr unsigned N_OUT = N_TILES; // every tile emits its own partial

inline std::string tile_in_file(const std::string &base, unsigned t) {
  const std::string key =
      ".n" + std::to_string(N_TILES) + "d" + std::to_string(DELTA);
  const auto dot = base.rfind('.');
  const std::string tag = key + ".t" + std::to_string(t);
  return dot == std::string::npos
             ? base + tag
             : base.substr(0, dot) + tag + base.substr(dot);
}

} // namespace bdtmt

class theGraph : public graph {
private:
  kernel k[bdtmt::N_TILES];

  void configure(kernel &kk) {
    source(kk) = "src/bdt_qs_obl.cpp";
    runtime<ratio>(kk) = 1.0;

    constexpr unsigned KIB = 1024;
    constexpr unsigned TABLES =
        sizeof(bdtm::QT_THR) + sizeof(bdtm::QT_BV) + sizeof(bdtm::QT_FEAT) +
        sizeof(bdtm::LEAVES) + sizeof(bdtm::INIT_V) + sizeof(bdtm::QT_BTERM) +
        sizeof(bdtm::QT_BSIGN) + sizeof(bdtm::BASIS_I) + sizeof(bdtm::BASIS_J) +
        sizeof(bdtm::BASIS_WI) + sizeof(bdtm::BASIS_WJ);
    // The basis lives on the stack, one vector per entry. Tree-split does not
    // make it smaller.
    constexpr unsigned XBYTES =
        (bdtm::N_FEATURES + bdtm::BASIS_N) * BDT_W * sizeof(bdtm::feat_t);
    heap_size(kk) = ((TABLES + KIB - 1) / KIB) * KIB + 2 * KIB;
    stack_size(kk) = ((XBYTES + KIB - 1) / KIB) * KIB + 4 * KIB;
  }

public:
  input_plio xin[bdtmt::N_IN];
  output_plio sout[bdtmt::N_OUT];

  theGraph() {
#if BDT_SPLIT_TREE && BDT_N_TILES > 1
#define BDT_LADDER_CREATE
#include "tile_roles.h"
#undef BDT_LADDER_CREATE
#else
    for (unsigned i = 0; i < bdtmt::N_TILES; i++)
      k[i] = kernel::create(bdt_qs_obl);
#endif
    for (unsigned i = 0; i < bdtmt::N_TILES; i++)
      configure(k[i]);

    for (unsigned i = 0; i < bdtmt::N_IN; i++) {
      const std::string name = "xin" + std::to_string(i);
      const std::string file = (bdtmt::SPLIT_TREE || bdtmt::N_TILES == 1)
                                   ? std::string(XIN_FILE)
                                   : bdtmt::tile_in_file(XIN_FILE, i);
      xin[i] = input_plio::create(name.c_str(), plio_64_bits, file.c_str(),
                                  bdtmt::PLIO_RATE);
    }
    for (unsigned i = 0; i < bdtmt::N_OUT; i++) {
      const std::string name = i == 0 ? "scores" : "scores" + std::to_string(i);
      const std::string file =
          i == 0 ? "scores.dat" : "scores.t" + std::to_string(i) + ".dat";
      sout[i] = output_plio::create(name.c_str(), plio_32_bits, file.c_str(),
                                    bdtmt::PLIO_RATE);
    }

    for (unsigned i = 0; i < bdtmt::N_TILES; i++) {
      connect<stream>(xin[bdtmt::SPLIT_TREE ? 0 : i].out[0], k[i].in[0]);
      connect<stream>(k[i].out[0], sout[i].in[0]);
    }
  }
};

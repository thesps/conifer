#pragma once
#include "parameters.h"
#include <aie_api/aie.hpp>
#include <type_traits>

#ifndef BDT_W
#define BDT_W 16
#endif

namespace bdtv {

constexpr unsigned W = BDT_W;

using feat_t = bdtm::feat_t;
using score_t = bdtm::score_t;
using bv_t = bdtm::bv_t;

using vfeat = aie::vector<feat_t, W>;
using vscore = aie::vector<score_t, W>;
using vidx = aie::vector<int16_t, W>;
using vmask = aie::mask<W>;

template <typename A>
using elem_of =
    std::remove_cv_t<std::remove_reference_t<decltype(std::declval<A &>()[0])>>;

using acc_tag =
    std::conditional_t<std::is_floating_point_v<score_t>, accfloat, acc32>;
using vacc = aie::accum<acc_tag, W>;

} // namespace bdtv
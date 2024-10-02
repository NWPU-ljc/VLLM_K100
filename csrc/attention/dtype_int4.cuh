#pragma once

#include "attention_generic.cuh"

#include <stdint.h>

namespace vllm {
#ifdef ENABLE_INT4
// int8 vector types for quantization of kv cache

template<>
struct Vec<int8_t, 1> {
    using Type = int8_t;
};

template<>
struct Vec<int8_t, 2> {
    using Type = int16_t;
};

template<>
struct Vec<int8_t, 4> {
    using Type = uint32_t;
};

template<>
struct Vec<int8_t, 8> {
    using Type = uint2;
};
#endif // ENABLE_FP8_E5M2

} // namespace vllm

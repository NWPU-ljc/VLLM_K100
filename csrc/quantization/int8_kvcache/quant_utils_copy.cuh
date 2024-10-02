#pragma once

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include "../../attention/attention_dtypes.h"
#include "../../attention/dtype_float32.cuh"
#include "../../attention/dtype_float16.cuh"
// #include "../../attention/dtype_bfloat16.cuh"

#include <cuda_fp16.h>

namespace vllm {
#ifdef ENABLE_INT8
namespace int8_unscaled {

template<typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x)
{
    return x;
}



// int8 -> half
template<>
__inline__ __device__ uint16_t vec_conversion<uint16_t, int8_t>(const int8_t& a)
{
    // 首先将 int8 转换为 float
    float temp = static_cast<float>(a);
    // 然后将 float 转换为 half
    return __float2half(temp);
}

// int8x2 -> half2
template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, int16_t>(const int16_t& a)
{
    union {
        uint16_t u16[2];
        uint32_t u32;
    } tmp;

    // 直接将 int16_t 的低8位和高8位转换为两个 int8_t
    int8_t int8_1 = static_cast<int8_t>(a & 0xFF); // 低8位
    int8_t int8_2 = static_cast<int8_t>((a >> 8) & 0xFF); // 高8位

    // 将 int8_t 转换为 float
    float float_1 = static_cast<float>(int8_1);
    float float_2 = static_cast<float>(int8_2);

    // 将 float 转换为 half
    __half half_1 = __float2half(float_1);
    __half half_2 = __float2half(float_2);

    // 将两个 half 组合成 half2
    __half2 res = __halves2half2(half_1, half_2);

    // 将 half2 转换为两个 uint16_t
    tmp.u16[0] = reinterpret_cast<uint16_t*>(&res)[0];
    tmp.u16[1] = reinterpret_cast<uint16_t*>(&res)[1];

    return tmp.u32;
}

// int8x4 -> half2x2
template<>
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t>(const uint32_t& a)
{
    union {
        uint2    u32x2;   // 用于返回两个 32 位的 half2 类型
        uint32_t u32[2];  // 用于存储中间结果
    } tmp;

    // 分解 uint32_t 为两个 uint16_t，分别包含 2 个 int8_t 值
    uint16_t lower_half = static_cast<uint16_t>(a & 0xFFFF);          // 提取低16位
    uint16_t upper_half = static_cast<uint16_t>((a >> 16) & 0xFFFF);  // 提取高16位

    // 使用已实现的 int8x2 转换为 half2 的函数
    tmp.u32[0] = vec_conversion<uint32_t, uint16_t>(lower_half);  // 转换低 2 个 int8
    tmp.u32[1] = vec_conversion<uint32_t, uint16_t>(upper_half);  // 转换高 2 个 int8

    return tmp.u32x2;
}

// int8x8 -> half2x4
template<>
__inline__ __device__ uint4 vec_conversion<uint4, uint2>(const uint2& a)
{
    union {
        uint4 u64x2;   // 用于返回四个 32 位的 half2 类型
        uint2 u64[2];  // 用于存储中间结果，每个 uint2 包含两个 32 位整数
    } tmp;

    // 将 uint2 分解为两个 uint32_t，分别包含 4 个 int8_t 值
    uint32_t lower = a.x;  // 包含低 4 个 int8_t
    uint32_t upper = a.y;  // 包含高 4 个 int8_t

    // 调用先前实现的 int8x4 转换为 half2x2 的函数
    tmp.u64[0] = vec_conversion<uint2, uint32_t>(lower);  // 转换低 4 个 int8_t
    tmp.u64[1] = vec_conversion<uint2, uint32_t>(upper);  // 转换高 4 个 int8_t

    return tmp.u64x2;
}

// int8 -> __nv_bfloat16
template<>
__inline__ __device__ __nv_bfloat16 vec_conversion<__nv_bfloat16, int8_t>(const int8_t& a)
{
    // int8 -> float -> bf16
    float tmp = static_cast<float>(a);

    // 将 float 转换为 __nv_bfloat16
    return __float2bfloat16(tmp);
}

// int8x2 -> __nv_bfloat162
template<>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, int16_t>(const int16_t& a)
{
    __nv_bfloat162 res;
    // 提取低8位并转换为 int8_t
    int8_t low = static_cast<int8_t>(a & 0xFF);
    
    // 提取高8位并转换为 int8_t
    int8_t high = static_cast<int8_t>((a >> 8) & 0xFF);
    
    // 转换每个 int8_t 到 __nv_bfloat16
    res.x = vec_conversion<__nv_bfloat16, int8_t>(low);
    res.y = vec_conversion<__nv_bfloat16, int8_t>(high);
    
    return res;
}

// int8x4 -> bf16_4_t
template<>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, uint32_t>(const uint32_t& a)
{
    bf16_4_t res;
    // 提取两个 int8x2 (每个 int8x2 用 int16_t 表示)
    int16_t low_half = static_cast<int16_t>(a & 0xFFFF);       // 低16位对应第一个 int8x2
    int16_t high_half = static_cast<int16_t>((a >> 16U) & 0xFFFF); // 高16位对应第二个 int8x2

    // 使用已有的 vec_conversion 将 int8x2 转换为 __nv_bfloat162
    res.x = vec_conversion<__nv_bfloat162, int16_t>(low_half);
    res.y = vec_conversion<__nv_bfloat162, int16_t>(high_half);

    return res;
}

// int8x8 -> bf16_8_t
template<>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, uint2>(const uint2& a)
{
    bf16_4_t tmp1, tmp2;
    // 使用 vec_conversion 将每个 uint32_t 转换为 bf16_4_t
    tmp1 = vec_conversion<bf16_4_t, uint32_t>(a.x); // 将 a.x (即 int8x4) 转换为 bf16_4_t
    tmp2 = vec_conversion<bf16_4_t, uint32_t>(a.y); // 将 a.y (即 int8x4) 转换为 bf16_4_t

    bf16_8_t res;

    // 将两个 bf16_4_t 合并为 bf16_8_t
    res.x = tmp1.x; // bf16_4_t 的第一个 __nv_bfloat162
    res.y = tmp1.y; // bf16_4_t 的第二个 __nv_bfloat162
    res.z = tmp2.x; // bf16_4_t 的第一个 __nv_bfloat162
    res.w = tmp2.y; // bf16_4_t 的第二个 __nv_bfloat162

    return res;
}

// int8 -> float
template<>
__inline__ __device__ float vec_conversion<float, int8_t>(const int8_t& a)
{
    return static_cast<float>(a);
}

// int8x2 -> float2
template<>
__inline__ __device__ float2 vec_conversion<float2, int16_t>(const int16_t& a)
{
    // 拆分 int16_t 为两个 int8_t
    int8_t low = static_cast<int8_t>(a & 0xFF);        // 低 8 位
    int8_t high = static_cast<int8_t>((a >> 8) & 0xFF); // 高 8 位

    // 将 int8_t 转换为 float
    float f1 = static_cast<float>(low);
    float f2 = static_cast<float>(high);

    // 构造 float2
    return make_float2(f1, f2);
}

// int8x4 -> float4
template<>
__inline__ __device__ Float4_ vec_conversion<Float4_, uint32_t>(const uint32_t& a)
{
    Float4_ res;

    // 将 uint32_t 拆分成两个 uint16_t
    uint16_t low = static_cast<int16_t>(a & 0xFFFF);         // 提取最低的 16 位（int8x2）
    uint16_t high = static_cast<int16_t>((a >> 16U) & 0xFFFF); // 提取最高的 16 位（int8x2）

    // 使用已实现的函数将 int8x2 转换为 float2
    res.x = vec_conversion<float2, int16_t>(low);  // 转换低 16 位
    res.y = vec_conversion<float2, int16_t>(high); // 转换高 16 位

    return res;
}

// int8x8 -> float8
template<>
__inline__ __device__ Float8_ vec_conversion<Float8_, uint2>(const uint2& a)
{
    Float4_ tmp1, tmp2;

    // 使用已实现的函数将 int8x4 转换为 float4
    tmp1 = vec_conversion<Float4_, uint32_t>(a.x);  // 处理前 4 个 int8_t
    tmp2 = vec_conversion<Float4_, uint32_t>(a.y);  // 处理后 4 个 int8_t

    // 构造 Float8_ 类型
    Float8_ res;
    res.x = tmp1.x;
    res.y = tmp1.y;
    res.z = tmp2.x;
    res.w = tmp2.y;

    return res;
}


// half -> int8
template<>
__inline__ __device__ int8_t vec_conversion<int8_t, uint16_t>(const uint16_t& a)
{
    // 创建 __half_raw 类型的临时变量
    __half_raw tmp;
    tmp.x = a;

    // 将 __half_raw 转换为 __half 类型，然后再转换为 float
    float tmp_float = __half2float(*reinterpret_cast<__half*>(&tmp));

    // 将 float 转换为 int8_t，进行四舍五入并处理溢出
    int8_t res = static_cast<int8_t>(roundf(fmaxf(fminf(tmp_float, 127.0f), -128.0f)));

    return res;
}

// bf16 -> int8
template<>
__inline__ __device__ int8_t vec_conversion<int8_t, __nv_bfloat16>(const __nv_bfloat16& a)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    assert(false);
#else
    // 将 __nv_bfloat16 转换为 float
    float tmp_float = __bfloat162float(a);

    // 将 float 转换为 int8_t，进行四舍五入并处理溢出
    int8_t res = static_cast<int8_t>(roundf(fmaxf(fminf(tmp_float, 127.0f), -128.0f)));

    return res;
#endif
}

// float -> int8
template<>
__inline__ __device__ int8_t vec_conversion<int8_t, float>(const float& a)
{
    // 将 float 值限制在 int8 的范围内，并进行四舍五入
    float clamped = fmaxf(fminf(a, 127.0f), -128.0f);
    int8_t res = static_cast<int8_t>(roundf(clamped));

    return res;
}

// int8x4 -> float4
template<>
__inline__ __device__ float4 vec_conversion<float4, uint32_t>(const uint32_t& a)
{
    Float4_ tmp = vec_conversion<Float4_, uint32_t>(static_cast<uint32_t>(a));
    // 将 Float4_ 中的 float2 合并为 float4
    float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
    return res;
}


template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, float2>(const float2& a)
{
    union {
        half2    float16;
        uint32_t uint32;
    };

    float16 = __float22half2_rn(a);
    return uint32;
}

template<>
__inline__ __device__ uint2 vec_conversion<uint2, Float4_>(const Float4_& a)
{
    uint2  b;
    float2 val;
    val.x = a.x.x;
    val.y = a.x.y;
    b.x   = vec_conversion<uint32_t, float2>(val);

    val.x = a.y.x;
    val.y = a.y.y;
    b.y   = vec_conversion<uint32_t, float2>(val);

    return b;
}

template<>
__inline__ __device__ float4 vec_conversion<float4, Float4_>(const Float4_& a)
{
    float4 b;
    b.x = a.x.x;
    b.y = a.x.y;
    b.z = a.y.x;
    b.w = a.y.y;
    return b;
}

template<>
__inline__ __device__ uint4 vec_conversion<uint4, Float8_>(const Float8_& a)
{
    uint4 b;
    b.x = vec_conversion<uint32_t, float2>(a.x);
    b.y = vec_conversion<uint32_t, float2>(a.y);
    b.z = vec_conversion<uint32_t, float2>(a.z);
    b.w = vec_conversion<uint32_t, float2>(a.w);
    return b;
}

template<>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, float2>(const float2 &a) {
    __nv_bfloat162 b;
    from_float(b, a);
    return b;
}

template<>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, Float4_>(const Float4_ &a) {
    bf16_4_t b;
    from_float(b, a);
    return b;
}

template<>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, Float8_>(const Float8_ &a) {
    bf16_8_t b;
    from_float(b, a);
    return b;
}

} // namespace fp8_e5m2_unscaled
#endif // ENABLE_FP8_E5M2
} // namespace vllm

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

namespace vllm {

// Activation and gating kernel template.
template<typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void act_and_mul_kernel(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., 2, d]
  const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = ACT_FN(x) * y;
  }
}

template<typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), int VEC>
__global__ void act_and_mul_kernel_opt1(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., 2, d]
  const int d) {
  // using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
  using VecType = aligned_vector<scalar_t, VEC>;
  const int64_t token_idx = blockIdx.x;
  int idx = threadIdx.x * VEC;
  if (idx < d) {
    const int64_t x_index = token_idx * 2 * d + idx;
    const int64_t y_index = token_idx * d + idx;
    VecType* x1 = (VecType*)(input + x_index);
    VecType* x2 = (VecType*)(input + x_index + d);
    VecType* y = (VecType*)(out + y_index);
    scalar_t r_x1[VEC];
    scalar_t r_x2[VEC];
    scalar_t r_y[VEC];
    *(VecType*)r_x1 = *x1;
    *(VecType*)r_x2 = *x2;
#pragma unroll
    for (int i = 0; i < VEC; i++) {
      r_y[i] = ACT_FN(r_x1[i]) * r_x2[i];
    }
    *y = *(VecType*)r_y;
  }
}

template<typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), int VEC>
__global__ void act_and_mul_kernel_opt2(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., 2, d]
  const int d) {
  // using VecType = at::native::memory::aligned_vector<scalar_t, VEC>;
  using VecType = aligned_vector<scalar_t, VEC>;
  const int64_t token_idx = blockIdx.x;
  int idx = threadIdx.x * VEC;
  for (; idx < d; idx += blockDim.x * VEC) {
    const int64_t x_index = token_idx * 2 * d + idx;
    const int64_t y_index = token_idx * d + idx;
    VecType* x1 = (VecType*)(input + x_index);
    VecType* x2 = (VecType*)(input + x_index + d);
    VecType* y = (VecType*)(out + y_index);
    scalar_t r_x1[VEC];
    scalar_t r_x2[VEC];
    scalar_t r_y[VEC];
    *(VecType*)r_x1 = *x1;
    *(VecType*)r_x2 = *x2;
#pragma unroll
    for (int i = 0; i < VEC; i++) {
      r_y[i] = ACT_FN(r_x1[i]) * r_x2[i];
    }
    *y = *(VecType*)r_y;
  }
}

template<typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T) (((float) x) / (1.0f + expf((float) -x)));
}

template<typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L38
  const float f = (float) x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T) (f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

} // namespace vllm

// Launch activation and gating kernel.
// #define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL)                                             \
//   int d = input.size(-1) / 2;                                                             \
//   int64_t num_tokens = input.numel() / input.size(-1);                                    \
//   dim3 grid(num_tokens);                                                                  \
//   dim3 block(std::min(d, 1024));                                                          \
//   const at::cuda::OptionalCUDAGuard device_guard(device_of(input));                       \
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                           \
//   VLLM_DISPATCH_FLOATING_TYPES(                                                           \
//     input.scalar_type(),                                                                  \
//     "act_and_mul_kernel",                                                                 \
//     [&] {                                                                                 \
//       vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>><<<grid, block, 0, stream>>>(   \
//         out.data_ptr<scalar_t>(),                                                         \
//         input.data_ptr<scalar_t>(),                                                       \
//         d);                                                                               \
//     });

#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL)                                             \
  int d = input.size(-1) / 2;                                                             \
  int64_t num_tokens = input.numel() / input.size(-1);                                    \
  dim3 grid(num_tokens);                                                                  \
  dim3 block(std::min(d, 1024));                                                          \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));                       \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                           \
  VLLM_DISPATCH_FLOATING_TYPES(                                                           \
    input.scalar_type(),                                                                  \
    "act_and_mul_kernel",                                                                 \
    [&] {                                                                                 \
    if (0 == d % 8 && d <= 16384) {                                                       \
      if (d <= 512) {                                                                     \
        vllm::act_and_mul_kernel_opt1<scalar_t, KERNEL<scalar_t>, 2>                      \
            <<<grid, 256, 0, stream>>>(out.data_ptr<scalar_t>(),                          \
                                       input.data_ptr<scalar_t>(), d);                    \
      } else if (d <= 1024) {                                                             \
        vllm::act_and_mul_kernel_opt1<scalar_t, KERNEL<scalar_t>, 8>                      \
            <<<grid, 128, 0, stream>>>(out.data_ptr<scalar_t>(),                          \
                                       input.data_ptr<scalar_t>(), d);                    \
      } else if (d <= 2048) {                                                             \
        vllm::act_and_mul_kernel_opt1<scalar_t, KERNEL<scalar_t>, 8>                      \
            <<<grid, 256, 0, stream>>>(out.data_ptr<scalar_t>(),                          \
                                       input.data_ptr<scalar_t>(), d);                    \
      } else if (d <= 4096) {                                                             \
        vllm::act_and_mul_kernel_opt1<scalar_t, KERNEL<scalar_t>, 8>                      \
            <<<grid, 512, 0, stream>>>(out.data_ptr<scalar_t>(),                          \
                                       input.data_ptr<scalar_t>(), d);                    \
      } else {                                                                            \
        vllm::act_and_mul_kernel_opt2<scalar_t, KERNEL<scalar_t>, 8>                      \
            <<<grid, 1024, 0, stream>>>(out.data_ptr<scalar_t>(),                         \
                                       input.data_ptr<scalar_t>(), d);                    \
      }                                                                                   \
    } else {                                                                              \
      vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>>                                \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),                        \
                                       input.data_ptr<scalar_t>(), d);                    \
    }                                                                                     \
    });

void silu_and_mul(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input)    // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel);
}

void gelu_and_mul(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input)    // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel);
}

namespace vllm {

// Element-wise activation kernel template.
template<typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., d]
  const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

} // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                                  \
  int d = input.size(-1);                                                                 \
  int64_t num_tokens = input.numel() / d;                                                 \
  dim3 grid(num_tokens);                                                                  \
  dim3 block(std::min(d, 1024));                                                          \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));                       \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                           \
  VLLM_DISPATCH_FLOATING_TYPES(                                                           \
    input.scalar_type(),                                                                  \
    "activation_kernel",                                                                  \
    [&] {                                                                                 \
      vllm::activation_kernel<scalar_t, KERNEL<scalar_t>><<<grid, block, 0, stream>>>(    \
        out.data_ptr<scalar_t>(),                                                         \
        input.data_ptr<scalar_t>(),                                                       \
        d);                                                                               \
    });

namespace vllm {

template<typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float) (x * x * x);
  const T t = (T) tanhf((T) (0.79788456f * (float) (x + (T) (0.044715f * x3))));
  return ((T) 0.5) * x * (((T) 1.0) + t);
}

template<typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float) x;
  const T t = (T) tanhf(((T) (f * 0.79788456f)) * (((T) 1.0) + (T) (0.044715f * f) * x));
  return ((T) 0.5) * x * (((T) 1.0) + t);
}

} // namespace vllm

void gelu_new(
  torch::Tensor& out,     // [..., d]
  torch::Tensor& input)   // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(
  torch::Tensor& out,     // [..., d]
  torch::Tensor& input)   // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

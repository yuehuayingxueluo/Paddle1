#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/core/tensor_utils.h"
#include <vector>
#include <assert.h>
#include "multi_tensor_apply_kernel.cuh"
#include "paddle/phi/kernels/multi_tensor_adam_kernel.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"

#include <cstdlib>

namespace phi {


#define BLOCK_SIZE 512
#define ILP 4

using MATH_T = float;

template<typename T, typename MT>
struct AdamFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    TensorListMetadata<5>& tl,
    MT beta1,
    MT beta2,
    const MT* beta1_pow_,
    const MT* beta2_pow_,
    MT epsilon,
    const MT* learning_rate,
    adamMode_t mode,
    bool multi_precision,
    MT decay)
  {

    MT lr = *learning_rate;
    MT beta1_pow = *beta1_pow_;
    MT beta2_pow = *beta2_pow_;
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;
    //获取这个block对应的tensor
    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    //获取这个block对应的chunk
    int chunk_idx = tl.block_to_chunk[blockIdx.x] + tl.start_chunk_this_tensor;
    //获取这个tensor对应的元素个数
    int n = tl.sizes[tensor_loc];
    //获取g的起始数据地址
    const T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;
    //获取p的起始数据地址
    MT* mp;
    T* p;
    p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;
    //获取m的起始数据地址
    MT* m = (MT*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;
    //获取v的起始数据地址
    MT* v = (MT*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;
    mp = (MT*)tl.addresses[4][tensor_loc];
    mp += chunk_idx*chunk_size;
    //需要计算的元素
    n -= chunk_idx*chunk_size;

    int flag = -1;

    // see note in multi_tensor_scale_kernel.cu
    //将数据分为ILP块
    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MT r_g[ILP];
      MT r_p[ILP];
      MT r_m[ILP];
      MT r_v[ILP];
      //每个线程处理每一块对应threadIdx.x的数据
      //向量化读取
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i == 46407){
          flag = ii;
        }
        if(i < n && i < chunk_size)
        {
          r_g[ii] = static_cast<MT>(g[i]);
          r_p[ii] = multi_precision ? mp[i] : static_cast<MT>(p[i]);
          r_m[ii] = static_cast<MT>(m[i]);
          r_v[ii] = static_cast<MT>(v[i]);
        } else {
          r_g[ii] = MT(0);
          r_p[ii] = MT(0);
          r_m[ii] = MT(0);
          r_v[ii] = MT(0);
        }
      }
      //进行具体的计算
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        if(mode == ADAM_MODE_0) {
          // r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          // r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          // r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          // MT next_m_unbiased = r_m[ii] / (1 - beta1_pow);
          // MT next_v_unbiased = r_v[ii] / (1 - beta2_pow);
          // MT denom = sqrtf(next_v_unbiased) + epsilon;
          // MT update = next_m_unbiased / denom;
          // r_p[ii] = r_p[ii] - (lr * update);
          if(chunk_idx == 2 && flag == ii){
            printf("i_start = %d, threadIdx.x = %d, blockIdx.x = %d\n", i_start, threadIdx.x, blockIdx.x);
            printf("multi_tensor_adam_kernel threadIdx.x = %d , p = %x, g = %x rm = %x, rv = %x, beta1 = %x, beta2 = %x\n", threadIdx.x, *(int *)&(r_p[ii]), *(int *)&(r_g[ii]), *(int *)&(r_m[ii]), *(int *)&(r_v[ii]), *(int *)&(beta1), *(int *)&(beta2));
            printf("multi_tensor_adam_kernel threadIdx.x = %d , p = %f, g = %f rm = %f, rv = %f, beta1 = %f, beta2 = %f\n", threadIdx.x, r_p[ii], r_g[ii], r_m[ii], r_v[ii], beta1, beta2);
          }
          r_m[ii] = beta1 * r_m[ii] + (static_cast<MT>(1.0)-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (static_cast<MT>(1.0)-beta2) * r_g[ii] * r_g[ii];
          if(chunk_idx == 2 && flag == ii){
            printf("1multi_tensor_adam_kernel threadIdx.x = %d , p = %x, g = %x rm = %x, rv = %x\n", threadIdx.x, *(int *)&(r_p[ii]), *(int *)&(r_g[ii]), *(int *)&(r_m[ii]), *(int *)&(r_v[ii]));
            printf("1multi_tensor_adam_kernel threadIdx.x = %d , p = %f, g = %f rm = %f, rv = %f\n", threadIdx.x, r_p[ii], r_g[ii], r_m[ii], r_v[ii]);
          }
          MT denom = (sqrt(r_v[ii]) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
          r_p[ii] += (r_m[ii] / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));
          if(chunk_idx == 2 && flag == ii){
            printf("2multi_tensor_adam_kernel threadIdx.x = %d , p = %x, g = %x rm = %x, rv = %x\n", threadIdx.x, *(int *)&(r_p[ii]), *(int *)&(r_g[ii]), *(int *)&(r_m[ii]), *(int *)&(r_v[ii]));
            printf("2multi_tensor_adam_kernel threadIdx.x = %d , p = %f, g = %f rm = %f, rv = %f\n", threadIdx.x, r_p[ii], r_g[ii], r_m[ii], r_v[ii]);
          }
        }
        else { // weight decay
          // r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          // r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          // MT next_m_unbiased = r_m[ii] / beta1_pow;
          // MT next_v_unbiased = r_v[ii] / beta2_pow;
          // MT denom = sqrtf(next_v_unbiased) + epsilon;
          // MT update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          // r_p[ii] = r_p[ii] - (lr * update);
          r_p[ii] *= (static_cast<MT>(1.0) - lr * decay);
          r_m[ii] = beta1 * r_m[ii] + (static_cast<MT>(1.0)-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (static_cast<MT>(1.0)-beta2) * r_g[ii] * r_g[ii];
          MT denom = (sqrt(r_v[ii]) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
          r_p[ii] += (r_m[ii] / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = static_cast<T>(r_p[ii]);
          m[i] = r_m[ii];
          v[i] = r_v[ii];
          if(multi_precision){
            mp[i] = r_p[ii];
          }
          if(chunk_idx == 2 && flag == ii){
            printf("1multi_tensor_adam_kernel threadIdx.x = %d , p = %f, rm = %f, rv = %f, mp = %f\n", threadIdx.x, p[i], m[i], v[i], mp[i]);
            flag = -1;
          }
        }
      }
    }
  }
};

template <typename T>
__global__ void UpdateBetaPow(T beta1,
                              T beta2,
                              const T* beta1_pow_,
                              const T* beta2_pow_,
                              T* beta1_pow_out,
                              T* beta2_pow_out) {
  *beta1_pow_out = beta1 * beta1_pow_[0];
  *beta2_pow_out = beta2 * beta2_pow_[0];
}

template <typename T,typename Context>
void MultiTensorAdamKernel(const Context& dev_ctx,
                           const std::vector<const DenseTensor *> &params,
                           const std::vector<const DenseTensor *> &grads,
                           const std::vector<const DenseTensor *> &moments1,
                           const std::vector<const DenseTensor *> &moments2,
                           const paddle::optional<std::vector<const DenseTensor*>>& master_param,
                           const DenseTensor &beta1_pow,
                           const DenseTensor &beta2_pow,
                           const DenseTensor &learning_rate,
                           const paddle::optional<DenseTensor>& skip_update,
                           const Scalar& beta1,
                           const Scalar& beta2,
                           const Scalar& epsilon,
                           int chunk_size,
                           float weight_decay,
                           bool mode,
                           bool multi_precision,
                           bool use_global_beta_pow,
                           std::vector<DenseTensor *> params_out,
                           std::vector<DenseTensor *> moments1_out,
                           std::vector<DenseTensor *> moments2_out,
                           std::vector<DenseTensor *> master_param_out,
                           DenseTensor *beta1_pow_out,
                           DenseTensor *beta2_pow_out) {

  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;
  MPDType beta1_ = beta1.to<MPDType>();
  MPDType beta2_ = beta2.to<MPDType>();
  MPDType weight_decay_ = static_cast<MPDType>(weight_decay);
  MPDType epsilon_ = epsilon.to<MPDType>();

  std::cout<<"params[0].dtype() = "<<params[0]->dtype()<<std::endl;

  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        skip_update->numel(),
        1,
        errors::InvalidArgument("Input(SkipUpdate) size must be 1, but get %d",
                                skip_update->numel()));
    std::vector<bool> skip_update_vec;
    paddle::framework::TensorToVector(*skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }

  // skip_update=true
  // mutable_data
  if (skip_update_) {
    VLOG(4) << "Adam skip update";
    return;
  }

  std::cout<<"1.cu"<<std::endl;

  std::vector<std::vector<DenseTensor *>> tensor_lists;

  tensor_lists.push_back(params_out);
  tensor_lists.push_back(moments1_out);
  tensor_lists.push_back(moments2_out);
  tensor_lists.push_back(master_param_out);

  std::cout<<"2.cu"<<std::endl;

  std::cout<<"(adamMode_t) mode"<<(adamMode_t) 0<<std::endl;

  multi_tensor_apply<5, MPDType>(
    dev_ctx,
    BLOCK_SIZE,
    chunk_size,
    tensor_lists,
    grads,
    AdamFunctor<T, MPDType>(),
    beta1_,
    beta2_,
    beta1_pow.data<MPDType>(),
    beta2_pow.data<MPDType>(),
    epsilon_,
    learning_rate.data<MPDType>(),
    mode ? (adamMode_t)1 : (adamMode_t)0,
    multi_precision,
    weight_decay_);

  cudaDeviceSynchronize();

  std::cout<<"static_cast<T>(0.617920) = "<<static_cast<T>(0.617920)<<std::endl;
  std::cout<<"T "<<typeid(T).name()<<std::endl;

  T *paramout_gpu = params_out[0]->data<T>();
  MPDType *masterparamout_gpu = master_param_out[0]->data<MPDType>();
  T *paramout_cpu = (T*) malloc((params_out[0]->numel())*sizeof(T));
  MPDType *masterparamout_cpu = (MPDType*) malloc((params_out[0]->numel())*sizeof(MPDType));

  for(int i = 0; i < params_out.size(); i++){
    //phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    //phi::Copy(dev_ctx, *(params_out[i]), CPUPlace(), false, (paramout[i]));
    cudaMemcpyAsync((void*)paramout_cpu, (void*)paramout_gpu, (params_out[0]->numel())*sizeof(T), cudaMemcpyDeviceToHost, dev_ctx.stream());
    cudaMemcpyAsync((void*)masterparamout_cpu, (void*)masterparamout_gpu, (params_out[0]->numel())*sizeof(MPDType), cudaMemcpyDeviceToHost, dev_ctx.stream());
  }
  
  std::cout<<"paramout_cpu[0] = "<<paramout_cpu[0]<<std::endl;
  std::cout<<"paramout_cpu[177479] = "<<*(paramout_cpu+177479)<<std::endl;
  std::cout<<"masterparamout_cpu[177479] = "<<*(masterparamout_cpu+177479)<<std::endl;

  if (!use_global_beta_pow) {
    // Update with gpu
    UpdateBetaPow<MPDType><<<1, 32, 0, dev_ctx.stream()>>>(
        beta1_,
        beta2_,
        beta1_pow.data<MPDType>(),
        beta2_pow.data<MPDType>(),
        dev_ctx.template Alloc<MPDType>(beta1_pow_out),
        dev_ctx.template Alloc<MPDType>(beta2_pow_out));
  }

  std::cout<<"3.cu"<<std::endl;

}

}  // namespace phi

PD_REGISTER_KERNEL(multi_tensor_adam,
                   GPU,
                   ALL_LAYOUT,
                   phi::MultiTensorAdamKernel,
                   phi::dtype::float16,
                   float,
                   double) {}



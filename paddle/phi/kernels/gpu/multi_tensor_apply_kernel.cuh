#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/core/tensor_utils.h"
#include <cuda_runtime.h>

#include <assert.h>

// #include <iostream>

// This header is the one-stop shop for all your multi-tensor apply needs.

namespace phi {

#define MAX_CHUNK_SIZE 65535

typedef enum{
  ADAM_MODE_0   =0, // L2 regularization mode
  ADAM_MODE_1   =1  // Decoupled weight decay mode(AdamW)
} adamMode_t;

// TODO:  Kernel arg size limit may be <4KB for some other cards (ie Jetson)
constexpr int depth_to_max_tensors[6] = {110, 64, 48, 36, 30, 24};
constexpr int depth_to_max_blocks[6] = {320, 320, 320, 320, 320, 320};

template<int n> struct TensorListMetadata
{
  const void* addresses[n][depth_to_max_tensors[n-1]];
  int sizes[depth_to_max_tensors[n-1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n-1]];
  //int16
  unsigned short int block_to_chunk[depth_to_max_blocks[n-1]]; // I fear this needs to be a full int.
  int start_tensor_this_launch;
  int start_chunk_this_tensor;
};


template<typename MT, typename T, typename U, typename... ArgTypes>
__global__ void multi_tensor_apply_kernel(
    int chunk_size,
    T tl,
    U callable,
    ArgTypes... args)
{
  // Hand the chunk information to the user-supplied functor to process however it likes.
  callable(chunk_size, tl, args...);
}

template<int depth, typename MT, typename FT, typename Context, typename... ArgTypes>
void multi_tensor_apply(
  const Context &dev_ctx,
  int block_size, //512
  int chunk_size, //2048*32
  const std::vector<std::vector<DenseTensor *>> &tensor_lists,
  const std::vector<const DenseTensor *> &g,
  FT callable,
  ArgTypes... args)
{
  std::cout<<"1"<<std::endl;
  PADDLE_ENFORCE_EQ(
        tensor_lists.size(),
        depth - 1,
        errors::InvalidArgument("ensor_lists.size() != depth - 1"));
  int len0 = tensor_lists[0].size();
  for(int  i = 0; i < depth - 1; i++){
    std::cout<<"tensor_lists[l].size() = "<<tensor_lists[i].size()<<","<<"len0 = "<<len0<<std::endl;
  }
  std::cout<<"2"<<std::endl;
  PADDLE_ENFORCE_GT(
      len0,
      0,
      errors::InvalidArgument(
          "tensor_lists[0].size() is not > 0"));
  auto ref_device = tensor_lists[0][0]->place();
  std::cout<<"3"<<std::endl;
  PADDLE_ENFORCE_NE(
        ref_device,
        CPUPlace(),
        errors::InvalidArgument("expected input to be on gpu"));
  std::cout<<"4"<<std::endl;
  for (int l = 0; l < tensor_lists.size(); l++) // No range-based for because I need indices
  {

    std::cout<<"tensor_lists[l].size() = "<<tensor_lists[l].size()<<","<<"len0 = "<<len0<<std::endl;

    PADDLE_ENFORCE_EQ(
        tensor_lists[l].size(),
        len0,
        errors::InvalidArgument("Size mismatch among tensor lists"));
    
    std::cout<<"tensor_lists[l].size() = "<<tensor_lists[l].size()<<","<<"len0 = "<<len0<<std::endl;

    for(int t = 0; t < tensor_lists[l].size(); t++)
    {
      // TODO:  Print which tensor fails.
//       bool contiguous_memory = tensor_lists[l][t].is_contiguous();
// #ifdef VERSION_GE_1_5
//       contiguous_memory = (contiguous_memory || tensor_lists[l][t].is_contiguous(at::MemoryFormat::ChannelsLast) || tensor_lists[l][t].is_contiguous(at::MemoryFormat::ChannelsLast3d));
// #endif
//       PADDLE_ENFORCE_EQ(
//         contiguous_memory,
//         true,
//         errors::InvalidArgument("A tensor was not contiguous."));
      PADDLE_ENFORCE_EQ(
        tensor_lists[l][t]->place(),
        ref_device,
        errors::InvalidArgument("A tensor was not on the same device as the first tensor"));
      PADDLE_ENFORCE_EQ(
        tensor_lists[l][t]->numel(),
        tensor_lists[0][t]->numel(),
        errors::InvalidArgument("Size mismatch"));
    }
  }

  std::cout<<"5"<<std::endl;

  //g中tensor的数目
  int ntensors = tensor_lists[0].size();

  std::cout<<"ntensors,"<<ntensors<<std::endl;

  TensorListMetadata<depth> tl;

  //const at::cuda::OptionalCUDAGuard device_guard(device_of(tensor_lists[0][0]));
  auto stream = dev_ctx.stream();
  //？ 从0号tensor开始
  tl.start_tensor_this_launch = 0;
  //当前block
  int loc_block_info = 0;
  //当前的tensor
  int loc_tensor_info = 0;
  for(int t = 0; t < ntensors; t++)
  {
    //获取kernel处理的对应tensor中的元素个数
    tl.sizes[loc_tensor_info] = tensor_lists[0][t]->numel();
    std::cout<<"tl.sizes[loc_tensor_info],"<<tl.sizes[loc_tensor_info]<<std::endl;
    //获取kernel处理的对应g p m v对应tensor的数据地址
    tl.addresses[0][loc_tensor_info] = g[t]->data();
    for(int d = 1; d < depth; d++)
      tl.addresses[d][loc_tensor_info] = tensor_lists[d - 1][t]->data();
    loc_tensor_info++;
    //这个tensor使用的chunk的数目
    int chunks_this_tensor = (tensor_lists[0][t]->numel() + chunk_size - 1)/chunk_size;
    std::cout<<"chunks_this_tensor,"<<chunks_this_tensor<<std::endl;
    tl.start_chunk_this_tensor = 0;
    int local_chunk = 0;

    cudaError_t error;

    for(int chunk = 0; chunk < chunks_this_tensor; chunk++)
    {
      // std::cout << chunks_this_tensor << std::endl;
      //block id对应的tensor
      tl.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      //block id对应的chunk
      //修改一下
      //第0个tensor的chunk id会超int16
      if(local_chunk > MAX_CHUNK_SIZE){
        tl.start_chunk_this_tensor += MAX_CHUNK_SIZE;
        local_chunk = 1;
      }
      tl.block_to_chunk[loc_block_info] = local_chunk;
      local_chunk++;
      loc_block_info++;
      //depth_to_max_tensors[depth-1] = 36
      //tensor已经填满
      bool tensors_full = (loc_tensor_info == depth_to_max_tensors[depth-1] &&
                           chunk == chunks_this_tensor - 1);
      //blcok已经填满
      bool blocks_full = (loc_block_info == depth_to_max_blocks[depth-1]);
      //最后一个chunk
      bool last_chunk = (t == ntensors - 1 && chunk == chunks_this_tensor - 1);
      if(tensors_full || blocks_full || last_chunk)
      {
        // using accscalar_t = acc_type<scalar_t, true>;
        multi_tensor_apply_kernel<MT><<<loc_block_info, block_size, 0, stream>>>(
          chunk_size, //2048*32
          tl,
          callable,
          args...);

        error = cudaGetLastError();
        std::cout<<"cudaGetErrorString "<<cudaGetErrorString(error)<<std::endl;
        
        // Reset.  The control flow possibilities here make my brain hurt.
        loc_block_info = 0;
        //如果这个tensor已经处理完
        if(chunk == chunks_this_tensor - 1)
        {
          // std::cout << "Hit case 1 " << cond1 << " " << cond2 << " " << cond3 << std::endl;
          loc_tensor_info = 0;
          //更新start_tensor_this_launch
          tl.start_tensor_this_launch = t + 1;
        }
        else
        {
          // std::cout << "Hit case 2 " << cond1 << " " << cond2 << " " << cond3 << std::endl;
          //将上一次处理tensor的元素个数赋值给sizes[0]
          tl.sizes[0] = tl.sizes[loc_tensor_info-1];
          //获取上一次处理tensor的地址
          for(int d = 0; d < depth; d++)
            tl.addresses[d][0] = tl.addresses[d][loc_tensor_info-1];
          loc_tensor_info = 1;
          tl.start_tensor_this_launch = t;
        }
      }
    }
  }
}

}  // namespace phi
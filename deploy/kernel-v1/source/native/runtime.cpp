#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#include <ATen/ops/_fused_adamw.h>
#include <ATen/ops/_foreach_zero.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstring>
#include <vector>

// Each process owns exactly one device/communicator. Rust's controller terminates
// the whole process group on a rank failure or operation deadline.
static thread_local std::string last_error;
extern "C" const char* jp_error() { return last_error.c_str(); }
extern "C" int jp_unique_id(unsigned char* bytes) {
  ncclUniqueId id; auto r=ncclGetUniqueId(&id);
  if(r!=ncclSuccess){last_error=ncclGetErrorString(r);return -1;}
  static_assert(sizeof(id)==128); std::memcpy(bytes,&id,128); return 0;
}
extern "C" void* jp_nccl_init(const unsigned char* bytes,int rank,int world) {
  auto cr=cudaSetDevice(rank);if(cr!=cudaSuccess){last_error=cudaGetErrorString(cr);return nullptr;}
  ncclUniqueId id;std::memcpy(&id,bytes,128); ncclComm_t comm;
  auto r=ncclCommInitRank(&comm,world,id,rank);
  if(r!=ncclSuccess){last_error=ncclGetErrorString(r);return nullptr;}return comm;
}
extern "C" int jp_allreduce(void* comm,torch::Tensor* t) {
  try {
    TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type()==torch::kFloat32,"NCCL expects contiguous CUDA FP32 tensors");
    auto stream=c10::cuda::getCurrentCUDAStream(t->get_device());
    auto r=ncclAllReduce(t->data_ptr<float>(),t->data_ptr<float>(),t->numel(),ncclFloat32,ncclSum,(ncclComm_t)comm,stream.stream());
    if(r!=ncclSuccess){last_error=ncclGetErrorString(r);return -1;}
    auto cr=cudaStreamSynchronize(stream.stream());if(cr!=cudaSuccess){last_error=cudaGetErrorString(cr);return -1;}
    return 0;
  } catch(const std::exception& e){last_error=e.what();return -1;}
}
extern "C" void jp_nccl_abort(void* comm){if(comm)ncclCommAbort((ncclComm_t)comm);}
extern "C" void jp_bf16(int enabled){
  // Match torch.autocast's scope lifetime. Cached BF16 parameter casts must not
  // survive an optimizer update to their FP32 masters.
  static thread_local std::vector<bool> previous;
  if(enabled){
    previous.push_back(at::autocast::is_autocast_enabled(at::kCUDA));
    at::autocast::increment_nesting();
    at::autocast::set_autocast_dtype(at::kCUDA,at::kBFloat16);
    at::autocast::set_autocast_enabled(at::kCUDA,true);
  } else {
    if(at::autocast::decrement_nesting()==0) at::autocast::clear_cache();
    bool restore=previous.back();previous.pop_back();
    at::autocast::set_autocast_enabled(at::kCUDA,restore);
  }
}

extern "C" int jp_fused_adamw(torch::Tensor** parameters, torch::Tensor** first,
                               torch::Tensor** second, size_t count,
                               int64_t step, double lr, double decay) {
  try {
    torch::NoGradGuard guard;
    std::vector<at::Tensor> p, g, m, v, steps;
    auto step_tensor=at::scalar_tensor(static_cast<double>(step),
                                     parameters[0]->options().dtype(at::kFloat));
    for(size_t i=0;i<count;i++) {
      p.push_back(*parameters[i]);g.push_back(parameters[i]->grad());
      m.push_back(*first[i]);v.push_back(*second[i]);steps.push_back(step_tensor);
    }
    at::_fused_adamw_(p,g,m,v,{},steps,lr,0.9,0.999,decay,1e-8,false,false);
    at::_foreach_zero_(g);
    return 0;
  } catch(const std::exception& e){last_error=e.what();return -1;}
}

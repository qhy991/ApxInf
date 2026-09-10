#pragma once
// BF16 moments and epsilon preserve the Torch 2.11 GroupNorm arithmetic contract.

struct GroupNormMoments { float mean, m2, count; };
__device__ GroupNormMoments group_norm_merge_moments(GroupNormMoments a, GroupNormMoments b) {
  const float n=a.count+b.count;
  if (n==0) return {0,0,0};
  const float delta=b.mean-a.mean;
  return {a.mean+delta*(b.count/n),a.m2+b.m2+delta*delta*(a.count*b.count/n),n};
}
__global__ void group_norm_bf16_rounded_kernel(const __nv_bfloat16* x,
    const __nv_bfloat16* w,const __nv_bfloat16* bias,__nv_bfloat16* out,
    int channels,int spatial,int groups,float eps) {
  const int group=blockIdx.x%groups;
  const int batch=blockIdx.x/groups;
  const int cpg=channels/groups;
  const int64_t width=static_cast<int64_t>(cpg)*spatial;
  const int64_t base=(static_cast<int64_t>(batch)*channels+group*cpg)*spatial;
  GroupNormMoments a{0,0,0};
  for(int64_t i=threadIdx.x;i<width;i+=blockDim.x) {
    const float v=__bfloat162float(x[base+i]);
    a=group_norm_merge_moments(a,{v,0,1});
  }
  __shared__ GroupNormMoments values[256];
  values[threadIdx.x]=a;
  __syncthreads();
  for(int offset=128;offset>0;offset>>=1) {
    if(threadIdx.x<offset) values[threadIdx.x]=group_norm_merge_moments(values[threadIdx.x],values[threadIdx.x+offset]);
    __syncthreads();
  }
  // Pinned Torch 2.11 stores the moments in the input dtype before affine.
  const float mean=__bfloat162float(__float2bfloat16(values[0].mean));
  const float rounded_eps=__bfloat162float(__float2bfloat16(eps));
  const float inv=__bfloat162float(__float2bfloat16(rsqrtf(values[0].m2/values[0].count+rounded_eps)));
  for(int64_t i=threadIdx.x;i<width;i+=blockDim.x) {
    const int c=group*cpg+i/spatial;
    const float scale=inv*__bfloat162float(w[c]);
    const float shift=__bfloat162float(bias[c])-mean*scale;
    const float v=__bfloat162float(x[base+i]);
    out[base+i]=__float2bfloat16(spatial==1
        ? fmaf((v-mean)*inv,__bfloat162float(w[c]),__bfloat162float(bias[c]))
        : fmaf(scale,v,shift));
  }
}

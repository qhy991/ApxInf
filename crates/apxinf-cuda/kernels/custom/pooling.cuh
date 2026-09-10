#pragma once
// NCHW 2x2 max pooling, stride 2, no padding, floor output dimensions.
__global__ void max_pool2x2_bf16_kernel(const __nv_bfloat16* x, __nv_bfloat16* y,
    int height, int width, int64_t count) {
  const int oh=height/2, ow=width/2;
  for (int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
    const int64_t plane=i/(oh*ow);
    const int py=(i/ow)%oh, px=i%ow;
    const int64_t base=plane*height*width+py*2*width+px*2;
    float v=__bfloat162float(x[base]);
    for (int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx) {
      const float n=__bfloat162float(x[base+dy*width+dx]);
      v=(isnan(v)||isnan(n)) ? nanf("") : fmaxf(v,n);
    }
    y[i]=__float2bfloat16(v);
  }
}

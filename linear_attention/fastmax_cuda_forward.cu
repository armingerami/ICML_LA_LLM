#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace {

// UNMASKED PART ////////////////////////////
template<typename scalar_t>
__global__
void calc_unmasked_cons_and_denum(const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d, int bhratio){
  extern __shared__ scalar_t s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  scalar_t tv, t;
  scalar_t tr[32];
  int sz = min(32,d);
  // int szr = 16; //number of reductions happeing in each thread; should be ~sqrt(d)
  // int szrb = 4; //number of reductions happeing in each thread; should be d/szr
  if(outer < d && i < bh){
    ikv = i/bhratio;
    // calc lin denum
    t = 0;
    for(int l = 0; l < nk; ++l){
      t += k[ikv][l][outer];
    }
    scalar_t a0div = nk/d;
    for(int l = 0; l < nq; ++l){
      atomicAdd(&o[i][l][d], q[i][l][outer]*t + a0div);
    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk;  ++l){
      t += v[ikv][l][outer];
    }
    for(int l = 0; l < nq;  ++l){
      o[i][l][outer] = t;
    }

  }
}

template<typename scalar_t>
__global__
void calc_unmasked_lin(const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ scalar_t s[];
  const int outer = threadIdx.x;
  const int mm = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  scalar_t tv, t;
  scalar_t tr[32];
  int sz = min(32,d);
  int szr = 4;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = 16; //number of reductions blocks; should be d/szr
  if(outer < d && mm < szrb && i < bh){
    ikv = i/bhratio;
    // calc lin
    for(int m = 0; m < szr; ++m) tr[m] = 0;
    for(int l = 0; l < nk;  ++l){
      tv = v[ikv][l][outer];
      s[outer+d] = k[ikv][l][outer];
      __syncthreads();
      for(int m = 0; m < szr; ++m){
        tr[m] += s[d+mm*szr+m]*tv;
      }
    }
    for(int l = 0; l < nq;  ++l){
      s[outer] = q[i][l][outer];
      __syncthreads();
      t = 0;
      for(int m = 0; m < szr; ++m){
        t += tr[m]*s[mm*szr+m];
      }
      atomicAdd(&o[i][l][outer],t);
    }
  }
}

// MASKED PART ////////////////////////////
template<typename scalar_t>
__global__
void calc_masked_cons_and_denum(const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ scalar_t s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  scalar_t tv, t;
  int ndiff = nk-nq;

  if(outer < d && i < bh){
    ikv = i/bhratio;
    // calc lin denum
    t = 0;
    for(int l = 0; l < nk-nq; ++l){
      t += k[ikv][l][outer];
    }
    scalar_t a0div = 1/d;
    int ndiff1 = ndiff+1;
    for(int l = 0; l < nq; ++l){
      t += k[ikv][ndiff+l][outer];
      atomicAdd(&o[i][l][d], q[i][l][outer]*t + a0div*(ndiff1+l));
    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk-nq; ++l){
      t += v[ikv][l][outer];
    }
    for(int l = 0; l < nq; ++l){
      t += v[ikv][ndiff+l][outer];
      o[i][l][outer] = t;
    }

  }
}

template<typename scalar_t>
__global__
void calc_masked_lin(const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ scalar_t s[];
  const int outer = threadIdx.x;
  const int mm = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  scalar_t tv, t;
  scalar_t tr[32];
  int sz = min(32,d);
  int szr = 4;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = 16; //number of reductions blocks; should be d/szr
  int ndiff = nk-nq;
  if(outer < d && mm < szrb && i < bh){
    ikv = i/bhratio;
    // calc lin
    for(int m = 0; m < szr; ++m) tr[m] = 0;
    for(int l = 0; l < nk-nq;  ++l){
      tv = v[ikv][l][outer];
      s[outer+d] = k[ikv][l][outer];
      __syncthreads();
      for(int m = 0; m < szr; ++m){
        tr[m] += s[d+mm*szr+m]*tv;
      }
    }
    for(int l = 0; l < nq;  ++l){
      tv = v[ikv][ndiff+l][outer];
      s[outer+d] = k[ikv][ndiff+l][outer];
      s[outer] = q[i][l][outer];
      __syncthreads();
      t = 0;
      for(int m = 0; m < szr; ++m){
        int mmm = mm*szr+m;
        tr[m] += s[d+mmm]*tv;
        t += tr[m]*s[mmm];
      }
      atomicAdd(&o[i][l][outer],t);
    }

  }
}

template<typename scalar_t>
__global__
void calc_div(torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> o, int bh, int nq, int d){
  const int outer = threadIdx.x;
  const int mm = blockIdx.x;
  const int i = blockIdx.y;
  int szr = 4;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = 16; //number of reductions blocks; should be d/szr
  if(outer < d && mm < szrb && i < bh){
    
    for(int l = mm; l < nq; l += szrb) o[i][l][outer] /= o[i][l][d];
  }
}

template<typename scalar_t>
__global__
void calc_norms(torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> norms, int bh, int n, int d, int th){
  const int ii = threadIdx.x;
  const int j = blockIdx.x;
  const int l = blockIdx.y;
  scalar_t t;
  int i;
  if(l < n && ii < th && j < ((bh-1)/th + 1)){
    i = j*th + ii;
    t = 0;
    for(int m = 0; m < d; m++){
      t += a[i][l][m]*a[i][l][m];
    }
    norms[i][l] = t;
  }
}

template<typename scalar_t>
__global__
void find_max(torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> norms, torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> maxes, int bh, int n, int th){
  const int ii = threadIdx.x;
  const int j = blockIdx.x;
  scalar_t t = 0;
  int i;
  if(ii < th && j < ((bh-1)/th + 1)){
    i = j*th + ii;
    for(int l = 0; l < n; ++l){
      t = max(t,norms[i][l]);
    }
    maxes[i] = t;
  }
}

template<typename scalar_t>
__global__
void apply_norm(torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> maxes, int bh, int n, int d, int n_seg){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  const int j = blockIdx.y;
  const int np = int(n/n_seg);
  scalar_t mx;
  if(m < d && i < bh){
    mx = maxes[i];
    if(mx < 0.1) mx = 0.1;
    for(int l = j*np; l < min(n,(j+1)*np); ++l){
      a[i][l][m] /= mx;
    }
  }
}


} // namespace

torch::Tensor forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool mask){

  return AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "forward_cuda", [&]() -> torch::Tensor {
    using scalar_t = float;

    const auto nq = q.size(1);
    const auto nk = k.size(1);
    const auto bh = q.size(0);
    const auto bhkv = k.size(0);
    const auto d = q.size(2);
    const int bhratio = bh/bhkv;

    const int threads = d; // threads = 256
    const int blocks = bh;

    const int n_seg = 128; // breaks context length into segments of n_seg, which are parallelized; i.e., paralleizes the code n_seg times
    // int szr = int(sqrt(d)); //number of reductions happeing in each thread; should be ~sqrt(d)
    // szr = 16;
    // int szrb = int(d/szr); //number of blocks performing reduction; should be ~sqrt(d)
    int szrb = 16;  //number of reduction blocks; should be ~sqrt(d)

    auto opts =  torch::TensorOptions().dtype(q.dtype()).layout(torch::kStrided).device(q.device());

    auto o = torch::zeros({bh,nq,d+1},opts);

    auto qnorms = torch::zeros({bh,nq},opts);
    auto knorms = torch::zeros({bh,nk},opts);
    auto qmaxes = torch::zeros({bh},opts);
    auto kmaxes = torch::zeros({bh},opts);



    const long th_lim = 1024;
    int th = min(th_lim, bh);
    calc_norms<<<dim3((bh-1)/th + 1, nq),th>>>(q.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),qnorms.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),bh,nq,d,th);
    find_max<<<(bh-1)/th + 1,th>>>(qnorms.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),bh,nk,th);
    for(int np = 0; np < int(nq/n_seg); ++np){
      apply_norm<<<dim3(blocks,n_seg),threads>>>(q.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),bh,nq,d,n_seg);
    }
    th = min(th_lim, bhkv);
    calc_norms<<<dim3((bhkv-1)/th + 1, nk),th>>>(k.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),knorms.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),bhkv,nk,d,th);
    find_max<<<(bhkv-1)/th + 1,th>>>(knorms.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),bhkv,nq,th);
    for(int np = 0; np < int(nk/n_seg); ++np){
      apply_norm<<<dim3(bhkv,n_seg),threads>>>(k.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),bhkv,nk,d,n_seg);
    }

    if(mask){
      calc_masked_cons_and_denum<<<blocks,threads,2*(d)*sizeof(scalar_t)>>>(q.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),k.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),v.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),o.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,bhratio);
      calc_masked_lin<<<dim3(szrb,blocks),threads,2*(d)*sizeof(scalar_t)>>>(q.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),k.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),v.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),o.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,bhratio);
    }
    else{
      calc_unmasked_cons_and_denum<<<blocks,threads,2*(d)*sizeof(scalar_t)>>>(q.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),k.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),v.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),o.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,bhratio);
      calc_unmasked_lin<<<dim3(szrb,blocks),threads,2*(d)*sizeof(scalar_t)>>>(q.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),k.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),v.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),o.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,bhratio);
    }
    calc_div<<<dim3(szrb,blocks),threads,2*(d)*sizeof(scalar_t)>>>(o.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),bh,nk,d);
    
    cudaDeviceSynchronize();

    // delete q;
    // delete k;
    // delete v;

    return o;
  });
}
//
// Created by lizhi on 2/11/23.
//
#include "mat.cu"
#define TILES_SM 8
#define WARPS (TN/TM)
using namespace nvcuda;
__device__ __forceinline__ void load_A_mat(float *vals, float *A, unsigned int num_bricks_to_load, unsigned int brick_loading_idx){
    unsigned int nnzs = num_bricks_to_load * TM * TK;
    for(unsigned int i = threadIdx.x; i < nnzs; i+=blockDim.x){
        A[i] = vals[i + brick_loading_idx*TM*TK];
    }
}
__device__ __forceinline__ void load_B_mat(unsigned int n_start, unsigned int k_start, unsigned int warp_id,
                           unsigned int lane_id, float *B, float *B_TILE, unsigned int K){
    for(unsigned int i = warp_id; i<TN; i+=WARPS){
        unsigned int n = i + n_start;
        unsigned int k_offset = k_start;
        #pragma unroll
        for(unsigned int j = lane_id; j<TB; j+=32){
            B_TILE[i*TB + j] = B[n*K+k_offset+j];
        }
    }
}

//load_B_mat(tn, unsigned int k_start, warp_id, lane_id, B, B_TILE, K);
__global__ void wmma_kernel(float *vals, float *B, unsigned int *bricksCount, unsigned int *brickPos, unsigned int *rowPtr,
                            unsigned int *cols, unsigned int N, unsigned int K, unsigned int MB, float * C){
    __shared__ float A_TILE[TILES_SM*TM*TK];
    __shared__ float B_TILE[TN*TB];
    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[TB/TM];

    for(unsigned int i = blockIdx.x; i<MB; i+=gridDim.x){
        unsigned int start = rowPtr[i];
        unsigned int end = rowPtr[i+1];
        if(start == end){
            continue;
        }
        for(unsigned int n = 0; n<N; n+=TN){
            #pragma unroll
            for(unsigned int ii=0;ii<TB/TM;++ii){
                wmma::fill_fragment(c_frag[ii], 0.0f);
            }
            for(unsigned int j=start; j<end; j++){
                unsigned int col = cols[j];
                load_B_mat(n, col * TB, warp_id, lane_id, B, B_TILE, K);
                unsigned int num_bricks = bricksCount[j];
                for(unsigned int ii=0;ii<num_bricks;ii+=TILES_SM){
                    unsigned int num_bricks_iter = min(TILES_SM, num_bricks - ii);
                    load_A_mat(&vals[j*TB*TB], A_TILE, num_bricks_iter, ii);
                    __syncthreads();
                    for(unsigned int jj=0; jj<num_bricks_iter; jj++){
                        unsigned int pos = brickPos[j*NUM_TK_BRICKS*NUM_TM_BRICKS + ii + jj];
                        unsigned int r = pos/NUM_TK_BRICKS;
                        unsigned int c = pos%NUM_TK_BRICKS;
                        c = c * TK;
                        wmma::load_matrix_sync(a_frag, A_TILE + jj*TM*TK, TK);
                        wmma::load_matrix_sync(b_frag, B_TILE + warp_id * TM * TB + c, TB);
                        #pragma unroll
                        for (unsigned t = 0; t < b_frag.num_elements; t++) {
                            b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
                        }
                        #pragma unroll
                        for (unsigned t = 0; t < a_frag.num_elements; t++) {
                            a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
                        }
                        wmma::mma_sync(c_frag[r], a_frag, b_frag, c_frag[r]);
                    }
                    __syncthreads();
                }
            }
            for(unsigned int ii=0;ii<TB/TM;++ii){
                wmma::store_matrix_sync(C + (i*TB)*N + ii*TM*N + n + warp_id*TM, c_frag[ii], N, wmma::mem_row_major);
            }
        }
    }
}
void check_gemm_correctness(float *c1, float *c2, unsigned int M, unsigned int N){
    float diff1 = 0.0f;
    float diff2 = 0.0f;
    float v1, v2;
    for(unsigned int i = 0;i<M*N; ++i){
        diff1 += abs(c1[i] - c2[i]);
	//std::cout<<c1[i]<<","<<c2[i]<<std::endl;
        if(max(abs(c1[i] - c2[i])/abs(c1[i]), abs(c1[i] - c2[i])/abs(c2[i])) > diff2){
            diff2 = max(abs(c1[i] - c2[i])/abs(c1[i]), abs(c1[i] - c2[i])/abs(c2[i]));
            v1 = c1[i];
            v2 = c2[i];
        }
    }
    std::cout<<"accumulated diff = "<<diff1<<" max diff single value "<<diff2<<" v1 vs v2 "<<v1<<","<<v2<<std::endl;
}

int main(int argc, char *argv[]){
  srand((unsigned int)time(NULL));
  unsigned int M = atoi(argv[1]);
  unsigned int N = atoi(argv[3]);
  unsigned int K = atoi(argv[2]);
  unsigned int nnz = atoi(argv[4]);
  std::string input_file = std::string(argv[5]);
  std::cout<<input_file<<std::endl;
  Mat mat(input_file,M,K,nnz);
  //float *a = new float[M*K];
  //read_graph(input_file, a, M, K);
  float *b = new float[K*N];
  //float *c = new float[M*N];
  float *BT = new float[K*N];
  std::cout<<"reach here"<<std::endl;
  for(unsigned int i=0;i<K*N;++i){
      b[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0f;
      //b[i] = 1.0f;
  }
  for(unsigned int i=0;i<K;++i){
     for(unsigned int j=0;j<N;++j){
       BT[j*K+i] = b[i*N+j];
     }
  }
  //gemm(a, b, c, M, K, N);
  float * c_device;
  chkerr(cudaMalloc(&c_device, M*N*sizeof(float)), __LINE__);
  chkerr(cudaMemset(c_device, 0, M*N*sizeof(float)), __LINE__);
  float * c_host = new float[M*N];
  float * b_device;
  chkerr(cudaMalloc(&b_device, K*N*sizeof(float)), __LINE__);
  chkerr(cudaMemcpy(b_device, BT, K*N*sizeof(float), cudaMemcpyHostToDevice), __LINE__);

  float temp_time;
  cudaEvent_t event_start;
  cudaEvent_t event_stop;
  float *vals = mat.valsDev;
  unsigned int * bricksCount = mat.activeBricksCountDev;
  unsigned int * brickPos = mat.activeBricksDev;
  unsigned int * rowPtr = mat.rowPtrDev;
  unsigned int * cols = mat.colsDev;
  unsigned int MB = mat.M;

  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  cudaEventRecord(event_start);
  wmma_kernel<<<432, WARPS * 32>>>(vals, b_device, bricksCount, brickPos, rowPtr, cols, N, K, MB, c_device);
  chkerr(cudaDeviceSynchronize(), __LINE__);
  cudaEventRecord(event_stop);
  cudaEventSynchronize(event_stop);
  cudaEventElapsedTime(&temp_time, event_start, event_stop);
  //float temp_time;
  chkerr(cudaMemcpy(c_host, c_device, M*N*sizeof(float), cudaMemcpyDeviceToHost), __LINE__);
  //check_gemm_correctness(c, c_host, M, N);
  std::cout<<"cuda time "<<temp_time<<std::endl;
  return 0;
}

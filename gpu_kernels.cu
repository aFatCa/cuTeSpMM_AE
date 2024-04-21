/*
How to run?
- ComputeType can be "__half" or "float"
$ export TEST=1; export ComputeType=float; export iters=50; export FILE_PATH=~/Repositories/SPMM_TensoreCore/data/small-scale/facebook_combined.mtx export n=128; export BlockSize=64; make ellblk;./ellBlk.o*/

#include "inc/CSR.hpp"

#include "inc/prepareBlkEll.hpp"
#include <bitset>
#include <cusparse.h>
#include "gespmm.h"

// sputnik includes
#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/spmm/cuda_spmm.h"
#include "sputnik/spmm/spmm_config.h"
#include <nvToolsExt.h>
#include <chrono>
#include <cuda_runtime.h>

#include "sputnik/test_utils.h"
#include "absl/random/random.h"
using namespace sputnik;


// Encapsule CUDA timing APIs.
//
// Usage:
//   GpuTimer timer; // create
//   timer.start();  // when you start recording
//   timer.stop();   // when  you stop recording
//   float dur = timer.elapsed_msecs(); // duration in milliseconds

size_t get_gpu_memory()
{
    int deviceId = 0; // Use the desired GPU device ID (0 for the first GPU)
    cudaSetDevice(deviceId);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    size_t totalMemory = deviceProp.totalGlobalMem;
    return totalMemory/1e9;
}
struct GpuTimer {
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
  
    GpuTimer() {
      cudaEventCreate(&startEvent);
      cudaEventCreate(&stopEvent);
    }
  
    ~GpuTimer() {
      cudaEventDestroy(startEvent);
      cudaEventDestroy(stopEvent);
    }
  
    void start() { cudaEventRecord(startEvent, 0); }
  
    void stop() {
      cudaEventRecord(stopEvent, 0);
      cudaEventSynchronize(stopEvent);
    }
  
    float elapsed_msecs() {
      float elapsed;
      cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
      return elapsed;
    }
  };
/*********************************** definitions **********************************/
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at file: %s line %d with error: %s (%d)\n",             \
               __FILE__,__LINE__, cudaGetErrorString(status), status);                 \
        return -1;                                                   \
    }                                                                          \
}
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at file %s line %d with error: %s (%d)\n",         \
              __FILE__, __LINE__, cusparseGetErrorString(status), status);              \
        return -1;                                                   \
    }                                                                          \
}

/*********************************** NVIDIA kernels **********************************/
static cudaDataType getCudaDataType(const float*)  { return CUDA_R_32F; }
static cudaDataType getCudaDataType(const __half*) { return CUDA_R_16F; }

float GPU_Compute_With_CuSparse_EllBlocked(COMPUTETYPE *hA_values, int* hA_columns, uint32_t A_EllColWidth,uint32_t A_EllColHeight, uint32_t A_ell_blocksize, COMPUTETYPE *hB, float *hC, uint32_t m, uint32_t k , uint32_t n, int warmup_iters,int iters, float alpha=1.0f, float beta=0.0f)
{
    // A, B, and C are row major;
    // cout<<"\n######################################################################\n";
    // cout<<"                     CuSparse_EllBlocked...\n";
    // cout<<"######################################################################\n";
    // NOTE: based on the https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-spmat-create-blockedell
    // ELL-Blocked columns: ellCols = blocksize * max_row_block_number
    // max_row_block_number is maximum number of nonempty blocks in all row blocks
    // The API cusparseDenseToSparse_convert just creates the ellValues and needs the ellCols from the user!! --> look CUDALibrarySamples/cuSPARSE/dense2sparse_blockedell
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    //                          Check compute capability
    //--------------------------------------------------------------------------
    cudaDeviceProp props;
    CHECK_CUDA( cudaGetDeviceProperties(&props, 0) )
    if (props.major < 7) {
      std::printf("cusparseSpMM with blocked ELL format is supported only "
                  "with compute capability at least 7.0\n");
      return -1;
    }
    //--------------------------------------------------------------------------
    //                          Device memory management
    //--------------------------------------------------------------------------

    uint32_t   A_num_rows      = m;
    uint32_t   A_num_cols      = k;
        
    uint32_t   B_num_rows      = k;
    uint32_t   B_num_cols      = n;
    uint32_t   ldb             = B_num_cols;             // B is row-major
    uint64_t   B_size          = B_num_rows * ldb;
    
    uint32_t   C_num_rows      = A_num_rows;
    uint32_t   C_num_cols      = B_num_cols;
    uint32_t   ldc             = C_num_cols;             // C is row-major
    uint64_t   C_size          = C_num_rows * ldc ;

    uint32_t    *dA_columns;
    COMPUTETYPE *dA_values, *dB;
    float  *dC;
    uint64_t A_num_blocks  = A_EllColWidth * A_EllColHeight;
    uint64_t A_Values_size = A_num_blocks  * A_ell_blocksize * A_ell_blocksize;
    uint64_t A_ell_cols    = (uint32_t) A_EllColWidth * A_ell_blocksize;            // ellCols
    // Device Memory Allocation
    int total_GPU_memory_needed = (A_num_blocks * sizeof(uint32_t))/1e9+( A_Values_size * sizeof(COMPUTETYPE))/1e9+(B_size * sizeof(COMPUTETYPE))/1e9+(C_size * sizeof(COMPUTETYPE))/1e9;
    if(get_gpu_memory()< total_GPU_memory_needed)
    {
        cout<<"Error: not enough GPU global memory!\n";
        cout<<"Memory requested:"<<total_GPU_memory_needed<<endl;
        cout<<"Memory available:"<<get_gpu_memory()<<endl;
        return -1;
    }

    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(uint32_t)))       // Output ellColInd 
    CHECK_CUDA( cudaMalloc((void**) &dA_values, A_Values_size * sizeof(COMPUTETYPE)))    // Output ellValue
    cout<<"dA_columns size:"<<(A_num_blocks * sizeof(uint32_t))/1e9<<"GB\n";
    cout<<"dA_values size:"<<( A_Values_size * sizeof(COMPUTETYPE))/1e9<<"GB\n";

    // Copy Data to Device
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_num_blocks * sizeof(uint32_t), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_Values_size * sizeof(COMPUTETYPE), cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    //                          CUSPARSE Wrapper APIs
    //--------------------------------------------------------------------------
    cusparseHandle_t     hndl = NULL;
    cusparseSpMatDescr_t matA_SP;
    CHECK_CUSPARSE( cusparseCreate(&hndl) )

    // Create sparse matrix A_SP in Blocked ELL format
    CHECK_CUSPARSE( cusparseCreateBlockedEll(
                                      &matA_SP,
                                      A_num_rows, A_num_cols, A_ell_blocksize,
                                      A_ell_cols, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, getCudaDataType(hA_values)) )
    // Device memory management
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(COMPUTETYPE)) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(float)) )
    cout<<"dB size:"<<(B_size * sizeof(COMPUTETYPE))/1e9<<"GB\n";
    cout<<"dC size:"<<(C_size * sizeof(COMPUTETYPE))/1e9<<"GB\n";
    CHECK_CUDA( cudaMemcpy(dB, hB, C_size * sizeof(float),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),cudaMemcpyHostToDevice) )
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB,getCudaDataType(hB), CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C_num_rows, C_num_cols, ldc, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW) )

    //--------------------------------------------------------------------------
    //                              perform matmul
    //--------------------------------------------------------------------------

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA_SP, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM warmup
    for(int iter = 0; iter< warmup_iters;iter++)
        CHECK_CUSPARSE( cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA_SP, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_BLOCKED_ELL_ALG1, dBuffer) )

    float elapsed_time = 0;
    cudaEvent_t start_event, stop_event ;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event);
    for(int iter = 0; iter< iters;iter++)
    {
        // execute SpMM
        CHECK_CUSPARSE( cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA_SP, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_BLOCKED_ELL_ALG1, dBuffer) )
    }
    cudaEventRecord(stop_event);
    CHECK_CUDA(cudaEventSynchronize(stop_event))
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event) ;
    //--------------------------------------------------------------------------
    //                    Write back the result + memory management
    //-------------------------------------------------------------------------
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),cudaMemcpyDeviceToHost) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA_SP) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    // device memory deallocation
    
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) ) 
    //--------------------------------------------------------------------------
    return elapsed_time/iters ;
}

float GPU_COO_cuSPARSE_NEW(int* hA_rows, int* hA_columns, COMPUTETYPE* hA_values, int nnz, COMPUTETYPE* hB, COMPUTETYPE* hC, uint32_t m,uint32_t k,uint32_t n, int warmup_iters,int iters, float alpha=1.0, float beta=0.0)
{
    // A, B, C are row-major
    // Host problem definition
    uint32_t   A_num_rows      = m;
    uint32_t   A_num_cols      = k;
    uint32_t   B_num_rows      = k;
    uint32_t   B_num_cols      = n;
    uint32_t   ldb             = B_num_cols;             // B is row-major
    uint64_t   B_size          = (uint64_t)B_num_rows * ldb;
    
    uint32_t   C_num_rows      = A_num_rows;
    uint32_t   C_num_cols      = B_num_cols;
    uint32_t   ldc             = C_num_cols;             // C is row-major
    uint64_t   C_size          = (uint64_t)C_num_rows * ldc ;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_rows, *dA_columns;
    COMPUTETYPE *dA_values, *dB;
    COMPUTETYPE  *dC;
    int total_GPU_memory_needed = ((uint64_t)nnz * sizeof(int))/1e9+
                                  ((uint64_t)nnz * sizeof(int))/1e9+
                                  ((uint64_t)nnz * sizeof(COMPUTETYPE))/1e9+
                                  (B_size * sizeof(COMPUTETYPE))/1e9+
                                  (C_size * sizeof(COMPUTETYPE))/1e9;
    if(get_gpu_memory()< total_GPU_memory_needed)
    {
        cout<<"Error: not enough GPU global memory!\n";
        cout<<"Memory requested:"<<total_GPU_memory_needed<<endl;
        cout<<"Memory available:"<<get_gpu_memory()<<endl;
        return -1;
    }

    CHECK_CUDA( cudaMalloc((void**) &dA_rows,    (uint64_t)nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, (uint64_t)nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  (uint64_t)nnz * sizeof(COMPUTETYPE))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(COMPUTETYPE)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(COMPUTETYPE)) )

    CHECK_CUDA( cudaMemcpy(dA_rows, hA_rows, (uint64_t)nnz * sizeof(int),
                            cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, (uint64_t)nnz * sizeof(int),
                            cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, (uint64_t)nnz * sizeof(COMPUTETYPE),
                            cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(COMPUTETYPE),
                            cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(COMPUTETYPE),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in COO format
    CHECK_CUSPARSE( cusparseCreateCoo(&matA, A_num_rows, A_num_cols, nnz,
                                        dA_rows, dA_columns, dA_values,
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, getCudaDataType(hA_values)) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB,getCudaDataType(hB), CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C_num_rows, C_num_cols, ldc, dC, getCudaDataType(hC), CUSPARSE_ORDER_ROW) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    
    // execute SpMM -- warp up
    for(int iter = 0; iter< warmup_iters;iter++)
        CHECK_CUSPARSE( cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    float elapsed_time = 0;
    cudaEvent_t start_event, stop_event ;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event);
    for(int iter = 0; iter< iters;iter++)
    {
        // execute SpMM
        CHECK_CUSPARSE( cusparseSpMM(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                        CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
    }
    cudaEventRecord(stop_event);
    CHECK_CUDA(cudaEventSynchronize(stop_event))
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event) ;
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(COMPUTETYPE),
                            cudaMemcpyDeviceToHost) )
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_rows) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return elapsed_time/iters;
}

float GPU_CSR_cuSPARSE_NEW(int* hA_csrOffsets, int*   hA_columns, COMPUTETYPE * hA_values, int A_nnz, COMPUTETYPE *hB, COMPUTETYPE *hC, uint32_t m, uint32_t k , uint32_t n, int warmup_iters,int iters, float alpha=1.0, float beta=0.0)
{
    // Host problem definition
    uint32_t   A_num_rows      = m;
    uint32_t   A_num_cols      = k;
    uint32_t   B_num_cols      = n;

    uint32_t   B_num_rows      = A_num_cols;             // row-major
    uint32_t   C_num_rows      = A_num_rows;
    uint32_t   C_num_cols      = B_num_cols;
    uint32_t   ldb             = B_num_cols;             // row-major
    uint32_t   ldc             = C_num_cols;             // row-major
    uint64_t   B_size          = (uint32_t)B_num_rows * ldb;
    uint64_t   C_size          = (uint32_t)C_num_rows * ldc;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    COMPUTETYPE* *dA_values, *dB;
    COMPUTETYPE  *dC;

    int total_GPU_memory_needed = ((uint64_t)(A_num_rows + 1) * sizeof(int))/1e9+
                                    ((uint32_t)A_nnz * sizeof(int))/1e9+
                                    ((uint32_t)A_nnz * sizeof(COMPUTETYPE))/1e9+
                                    (B_size * sizeof(COMPUTETYPE))/1e9+
                                    (C_size * sizeof(float))/1e9;
    if(get_gpu_memory()< total_GPU_memory_needed)
    {
        cout<<"Error: not enough GPU global memory!\n";
        cout<<"Memory requested:"<<total_GPU_memory_needed<<endl;
        cout<<"Memory available:"<<get_gpu_memory()<<endl;
        return -1;
    }
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,(uint64_t)(A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, (uint32_t)A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  (uint32_t)A_nnz * sizeof(COMPUTETYPE))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(COMPUTETYPE)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(COMPUTETYPE)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,(uint64_t)(A_num_rows + 1) * sizeof(int),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, (uint64_t)A_nnz * sizeof(int),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, (uint64_t)A_nnz * sizeof(COMPUTETYPE),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(COMPUTETYPE),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(COMPUTETYPE),cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, k, A_nnz,
                                    dA_csrOffsets, dA_columns, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO,  getCudaDataType(hA_values)))
    
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, k, n, n, dB, getCudaDataType(hA_values), CUSPARSE_ORDER_ROW) )
    
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, m, n, n, dC, getCudaDataType(hC), CUSPARSE_ORDER_ROW) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_CSR_ALG2, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // Execute SpMM warm up
    for(int iter = 0; iter< warmup_iters;iter++)
        CHECK_CUSPARSE( cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_CSR_ALG2, dBuffer) )

    // execute SpMM
    float elapsed_time = 0;
    cudaEvent_t start_event, stop_event ;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event);
    for(int iter = 0; iter< iters;iter++)
    {
        // Execute SpMM
        CHECK_CUSPARSE( cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_CSR_ALG2, dBuffer) )

    }
    cudaEventRecord(stop_event);
    CHECK_CUDA( cudaEventSynchronize(stop_event) )
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event) ;
                                
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    // CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // cpy results to host
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(COMPUTETYPE),cudaMemcpyDeviceToHost) )
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return elapsed_time/iters;
}

float GPU_gespmm(int* hA_csrOffsets, int*   hA_columns, COMPUTETYPE * hA_values, int A_nnz, COMPUTETYPE *hB, float *hC, int M, int K , int N, int warmup_iters,int iters)
{
    // int M;                               // number of A-rows
    // int K;                               // number of A-columns
    int nnz = A_nnz;                             // number of non-zeros in A
    // std::vector<int> csr_indptr_buffer;  // buffer for indptr array in CSR format
    // std::vector<int> csr_indices_buffer; // buffer for indices (column-ids) array
                                        // in CSR format
    // load sparse matrix from mtx file
    // read_mtx_file(argv[1], M, K, nnz, csr_indptr_buffer, csr_indices_buffer);
    // printf("Finish reading matrix %d rows, %d columns, %d nnz. \nIgnore original "
    //         "values and use randomly generated values.\n",
    //         M, K, nnz);

    // // Create GPU arrays
    // int N = 128; // number of B-columns
    // if (argc > 2) {
    //     N = atoi(argv[2]);
    // }
    // assert(
    //     N > 0 &&
    //     "second command-line argument is number of B columns, should be >0.\n");

    float  *C_h = hC ; //, *C_ref = NULL;
    COMPUTETYPE *B_h = hB, *csr_values_h = hA_values;
    float *B_d = NULL, *C_d = NULL, *csr_values_d = NULL;
    int *csr_indptr_d = NULL, *csr_indices_d = NULL;

    // B_h = (float *)malloc(sizeof(float) * K * N);
    // C_h = (float *)malloc(sizeof(float) * M * N);
    // C_ref = (float *)malloc(sizeof(float) * M * N);
    // csr_values_h = (float *)malloc(sizeof(float) * nnz);
    // if (!B_h || !C_h || !C_ref || !csr_values_h) {
    //     printf("Host allocation failed.\n");
    //     return EXIT_FAILURE;
    // }

    // fill_random(csr_values_h, nnz);
    // fill_random(B_h, K * N);
    int total_GPU_memory_needed = (sizeof(float) * (uint64_t)K * N)/1e9+
                                    (sizeof(float) * (uint64_t)M * N)/1e9+
                                    (sizeof(float) * (uint64_t)nnz)/1e9+
                                    (sizeof(int) * (uint64_t)(M + 1))/1e9+
                                    (sizeof(int) * (uint64_t)nnz)/1e9;
    if(get_gpu_memory()< total_GPU_memory_needed)
    {
        cout<<"Error: not enough GPU global memory!\n";
        cout<<"Memory requested:"<<total_GPU_memory_needed<<endl;
        cout<<"Memory available:"<<get_gpu_memory()<<endl;
        return -1;
    }

    CHECK_CUDA(cudaMalloc((void **)&B_d, sizeof(float) * (uint64_t)K * N));
    CHECK_CUDA(cudaMalloc((void **)&C_d, sizeof(float) * (uint64_t)M * N));
    CHECK_CUDA(cudaMalloc((void **)&csr_values_d, sizeof(float) * (uint64_t)nnz));
    CHECK_CUDA(cudaMalloc((void **)&csr_indptr_d, sizeof(int) * (uint64_t)(M + 1)));
    CHECK_CUDA(cudaMalloc((void **)&csr_indices_d, sizeof(int) * (uint64_t)nnz));

    CHECK_CUDA(cudaMemcpy(B_d, B_h, sizeof(float) * (uint64_t)K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(C_d, 0x0, sizeof(float) * (uint64_t)M * N));
    CHECK_CUDA(cudaMemcpy(csr_values_d, csr_values_h, sizeof(float) * (uint64_t)nnz,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csr_indptr_d, hA_csrOffsets,sizeof(int) * (uint64_t)(M + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csr_indices_d, hA_columns,sizeof(int) * (uint64_t)nnz, cudaMemcpyHostToDevice));

    SpMatCsrDescr_t spmatA{M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d};
    gespmmAlg_t algs[] = {
        GESPMM_ALG_SEQREDUCE_ROWBALANCE,  GESPMM_ALG_PARREDUCE_ROWBALANCE,
        GESPMM_ALG_SEQREDUCE_NNZBALANCE,  GESPMM_ALG_PARREDUCE_NNZBALANCE,
        GESPMM_ALG_ROWCACHING_ROWBALANCE, GESPMM_ALG_ROWCACHING_NNZBALANCE};
    
    float best_gspmm_ms = std::numeric_limits<float>::infinity();
    for (auto alg : algs) 
    {
        //
        // Run GE-SpMM and check result
        //
    
        CHECK_CUDA(cudaMemset(C_d, 0x0, sizeof(float) * (uint64_t)M * N));
        
        
        // spmm_reference_host<int, float>(M, N, K, csr_indptr_buffer.data(),
        // csr_indices_buffer.data(), csr_values_h,
        // B_h, C_ref);
        
        // bool correct = check_result<float>(M, N, C_h, C_ref);
        
        // if (correct) {
            // // benchmark GE-SpMM performance
            const char* algo_name;
            switch (alg) {
                case GESPMM_ALG_SEQREDUCE_ROWBALANCE:
                algo_name = "SEQREDUCE-ROWBALANCE";
                break;
                case GESPMM_ALG_PARREDUCE_ROWBALANCE:
                algo_name = "PARREDUCE-ROWBALANCE";
                break;
                case GESPMM_ALG_SEQREDUCE_NNZBALANCE:
                algo_name = "SEQREDUCE-NNZBALANCE";
                break;
                case GESPMM_ALG_PARREDUCE_NNZBALANCE:
                algo_name = "PARREDUCE-NNZBALANCE";
                break;
                case GESPMM_ALG_ROWCACHING_ROWBALANCE:
                algo_name = "ROWCACHING-ROWBALANCE";
                break;
                case GESPMM_ALG_ROWCACHING_NNZBALANCE:
                algo_name = "ROWCACHING-NNZBALANCE";
                break;
                default:
                printf("err: unknown gespmm alg");
                return -1;
                // assert(0); // This will trigger an assertion failure
            }
            
            GpuTimer gpu_timer;
            int warmup_iter = warmup_iters;
            int repeat_iter = iters;
            nvtxEventAttributes_t eventAttrib = {0};
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
            char message[100];
            sprintf(message, "gespmm_%s_%d", algo_name, N);
            eventAttrib.message.ascii = message;
            nvtxRangePushEx(&eventAttrib);
            for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) 
            {
                if (iter == warmup_iter) 
                gpu_timer.start();
                gespmmCsrSpMM(spmatA, B_d, N, C_d, true, alg);
            }
            nvtxRangePop();
            gpu_timer.stop();
            
            float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
            if(kernel_dur_msecs < best_gspmm_ms)
            best_gspmm_ms = kernel_dur_msecs;
            
        cudaDeviceSynchronize();
        
        
        // float MFlop_count = (float)nnz / 1e6 * N * 2;
        
        // float gflops = MFlop_count / kernel_dur_msecs;
        
        // printf("[GE-SpMM][Alg: %d] Report: spmm A(%d x %d) * B(%d x %d) sparsity "
        //         "%f (nnz=%d) \n Time %f (ms), Throughput %f (gflops).\n",
        //         alg, M, K, K, N, (float)nnz / M / K, nnz, kernel_dur_msecs,
        //         gflops);
        // }
    }
    CHECK_CUDA(cudaMemcpy(C_h, C_d, sizeof(float) * (uint64_t)M * N, cudaMemcpyDeviceToHost));
    // device memory deallocation
    CHECK_CUDA( cudaFree(B_d) )
    CHECK_CUDA( cudaFree(C_d) )
    CHECK_CUDA( cudaFree(csr_values_d) )
    CHECK_CUDA( cudaFree(csr_indptr_d) )
    CHECK_CUDA( cudaFree(csr_indices_d) )

    return best_gspmm_ms;
}


float GPU_gespmm_singleAlg(int* hA_csrOffsets, int*   hA_columns, COMPUTETYPE * hA_values, int A_nnz, COMPUTETYPE *hB, float *hC, int M, int K , int N, int warmup_iters,int iters, string algo_name)
{
    // int M;                               // number of A-rows
    // int K;                               // number of A-columns
    int nnz = A_nnz;                             // number of non-zeros in A
    // std::vector<int> csr_indptr_buffer;  // buffer for indptr array in CSR format
    // std::vector<int> csr_indices_buffer; // buffer for indices (column-ids) array
                                        // in CSR format
    // load sparse matrix from mtx file
    // read_mtx_file(argv[1], M, K, nnz, csr_indptr_buffer, csr_indices_buffer);
    // printf("Finish reading matrix %d rows, %d columns, %d nnz. \nIgnore original "
    //         "values and use randomly generated values.\n",
    //         M, K, nnz);

    // // Create GPU arrays
    // int N = 128; // number of B-columns
    // if (argc > 2) {
    //     N = atoi(argv[2]);
    // }
    // assert(
    //     N > 0 &&
    //     "second command-line argument is number of B columns, should be >0.\n");

    float  *C_h = hC; //, *C_ref = NULL;
    COMPUTETYPE *B_h = hB, *csr_values_h = hA_values;
    float *B_d = NULL, *C_d = NULL, *csr_values_d = NULL;
    int *csr_indptr_d = NULL, *csr_indices_d = NULL;

    // B_h = (float *)malloc(sizeof(float) * K * N);
    // C_h = (float *)malloc(sizeof(float) * M * N);
    // C_ref = (float *)malloc(sizeof(float) * M * N);
    // csr_values_h = (float *)malloc(sizeof(float) * nnz);
    // if (!B_h || !C_h || !C_ref || !csr_values_h) {
    //     printf("Host allocation failed.\n");
    //     return EXIT_FAILURE;
    // }

    // fill_random(csr_values_h, nnz);
    // fill_random(B_h, K * N);
    int total_GPU_memory_needed = (sizeof(float) * (uint64_t)K * N)/1e9+
                                    (sizeof(float) * (uint64_t)M * N)/1e9+
                                    (sizeof(float) * (uint64_t)nnz)/1e9+
                                    (sizeof(int) * (uint64_t)(M + 1))/1e9+
                                    (sizeof(int) * (uint64_t)nnz)/1e9;
    if(get_gpu_memory()< total_GPU_memory_needed)
    {
        cout<<"Error: not enough GPU global memory!\n";
        cout<<"Memory requested:"<<total_GPU_memory_needed<<endl;
        cout<<"Memory available:"<<get_gpu_memory()<<endl;
        return -1;
    }
    CHECK_CUDA(cudaMalloc((void **)&B_d, sizeof(float) * (uint64_t)K * N));
    CHECK_CUDA(cudaMalloc((void **)&C_d, sizeof(float) * (uint64_t)M * N));
    CHECK_CUDA(cudaMalloc((void **)&csr_values_d, sizeof(float) * (uint64_t)nnz));
    CHECK_CUDA(cudaMalloc((void **)&csr_indptr_d, sizeof(int) * (uint64_t)(M + 1)));
    CHECK_CUDA(cudaMalloc((void **)&csr_indices_d, sizeof(int) * (uint64_t)nnz));

    CHECK_CUDA(cudaMemcpy(B_d, B_h, sizeof(float) * (uint64_t)K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(C_d, 0x0, sizeof(float) * (uint64_t)M * N));
    CHECK_CUDA(cudaMemcpy(csr_values_d, csr_values_h, sizeof(float) * (uint64_t)nnz,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csr_indptr_d, hA_csrOffsets,sizeof(int) * (uint64_t)(M + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csr_indices_d, hA_columns,sizeof(int) * (uint64_t)nnz, cudaMemcpyHostToDevice));

    SpMatCsrDescr_t spmatA{M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d};
    // gespmmAlg_t algs[] = {
    //     GESPMM_ALG_SEQREDUCE_ROWBALANCE,  GESPMM_ALG_PARREDUCE_ROWBALANCE,
    //     GESPMM_ALG_SEQREDUCE_NNZBALANCE,  GESPMM_ALG_PARREDUCE_NNZBALANCE,
    //     GESPMM_ALG_ROWCACHING_ROWBALANCE, GESPMM_ALG_ROWCACHING_NNZBALANCE};
    // float best_gspmm_ms = std::numeric_limits<float>::infinity();
    // for (auto alg : algs) 
    // {
        //
        // Run GE-SpMM and check result
        //
    
        CHECK_CUDA(cudaMemset(C_d, 0x0, sizeof(float) * (uint64_t)M * N));
        
        
        // spmm_reference_host<int, float>(M, N, K, csr_indptr_buffer.data(),
        // csr_indices_buffer.data(), csr_values_h,
        // B_h, C_ref);
        
        // bool correct = check_result<float>(M, N, C_h, C_ref);
        
        // if (correct) {
            // // benchmark GE-SpMM performance
            gespmmAlg_t alg;

            if (algo_name == "gespmm_SEQREDUCE-ROWBALANCE") {
                alg = GESPMM_ALG_SEQREDUCE_ROWBALANCE;
            } else if (algo_name == "gespmm_PARREDUCE-ROWBALANCE") {
                alg = GESPMM_ALG_PARREDUCE_ROWBALANCE;
            } else if (algo_name == "gespmm_SEQREDUCE-NNZBALANCE") {
                alg = GESPMM_ALG_SEQREDUCE_NNZBALANCE;
            } else if (algo_name == "gespmm_PARREDUCE-NNZBALANCE") {
                alg = GESPMM_ALG_PARREDUCE_NNZBALANCE;
            } else if (algo_name == "gespmm_ROWCACHING-ROWBALANCE") {
                alg = GESPMM_ALG_ROWCACHING_ROWBALANCE;
            } else if (algo_name == "gespmm_ROWCACHING-NNZBALANCE") {
                alg = GESPMM_ALG_ROWCACHING_NNZBALANCE;
            } else {
                printf("err: unknown gespmm alg");
                assert(0); // This will trigger an assertion failure
            }

            GpuTimer gpu_timer;
            int warmup_iter = warmup_iters;
            int repeat_iter = iters;
            nvtxEventAttributes_t eventAttrib = {0};
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
            char message[100];
            sprintf(message, "%s_%d", algo_name.c_str(), N);
            eventAttrib.message.ascii = message;
            nvtxRangePushEx(&eventAttrib);
            for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) 
            {
                if (iter == warmup_iter) 
                gpu_timer.start();
                gespmmCsrSpMM(spmatA, B_d, N, C_d, true, alg);
            }
            nvtxRangePop();
            gpu_timer.stop();
            
            float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
            // if(kernel_dur_msecs < best_gspmm_ms)
            // best_gspmm_ms = kernel_dur_msecs;
            
        cudaDeviceSynchronize();
        
        
        // float MFlop_count = (float)nnz / 1e6 * N * 2;
        
        // float gflops = MFlop_count / kernel_dur_msecs;
        
        // printf("[GE-SpMM][Alg: %d] Report: spmm A(%d x %d) * B(%d x %d) sparsity "
        //         "%f (nnz=%d) \n Time %f (ms), Throughput %f (gflops).\n",
        //         alg, M, K, K, N, (float)nnz / M / K, nnz, kernel_dur_msecs,
        //         gflops);
        // }
    // }
    CHECK_CUDA(cudaMemcpy(C_h, C_d, sizeof(float) * (uint64_t)M * N, cudaMemcpyDeviceToHost));
    // device memory deallocation
    CHECK_CUDA( cudaFree(B_d) )
    CHECK_CUDA( cudaFree(C_d) )
    CHECK_CUDA( cudaFree(csr_values_d) )
    CHECK_CUDA( cudaFree(csr_indptr_d) )
    CHECK_CUDA( cudaFree(csr_indices_d) )

    return kernel_dur_msecs;
}


float GPU_sputnik(int* hA_csrOffsets, int* hA_columns, COMPUTETYPE* hA_values,int* A_row_permutation, int A_nnz, COMPUTETYPE *hB, float *hC, int M, int K , int N, int warmup_iters,int iters)
{
    int nnz = A_nnz;                             // number of non-zeros in A

    float  *C_h = hC;
    COMPUTETYPE *B_h = hB, *csr_values_h = hA_values;
    float *B_d = NULL, *C_d = NULL, *csr_values_d = NULL;
    int *csr_indptr_d = NULL, *csr_indices_d = NULL;
    int* A_row_permutation_d = NULL;
    int total_GPU_memory_needed = (sizeof(float) * (uint64_t)K * N)/1e9+
                                    (sizeof(float) * (uint64_t)M * N)/1e9+
                                    (sizeof(float) * (uint64_t)nnz)/1e9+
                                    (sizeof(int) * (uint64_t)(M + 1))/1e9+
                                    (sizeof(int) * (uint64_t)nnz)/1e9+
                                    (sizeof(int) * (uint64_t)M)/1e9;
    if(get_gpu_memory()< total_GPU_memory_needed)
    {
        cout<<"Error: not enough GPU global memory!\n";
        cout<<"Memory requested:"<<total_GPU_memory_needed<<endl;
        cout<<"Memory available:"<<get_gpu_memory()<<endl;
        return -1;
    }
    CHECK_CUDA(cudaMalloc((void **)&B_d, sizeof(float) * (uint64_t)K * N));
    CHECK_CUDA(cudaMalloc((void **)&C_d, sizeof(float) * (uint64_t)M * N));
    CHECK_CUDA(cudaMalloc((void **)&csr_values_d, sizeof(float) * (uint64_t)nnz));
    CHECK_CUDA(cudaMalloc((void **)&csr_indptr_d, sizeof(int) * (uint64_t)(M + 1)));
    CHECK_CUDA(cudaMalloc((void **)&csr_indices_d, sizeof(int) * (uint64_t)nnz));
    CHECK_CUDA(cudaMalloc((void **)&A_row_permutation_d, sizeof(int) * (uint64_t)M));

    CHECK_CUDA(cudaMemcpy(B_d, B_h, sizeof(float) * (uint64_t)K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(C_d, 0x0, sizeof(float) * (uint64_t)M * N));
    CHECK_CUDA(cudaMemcpy(csr_values_d, csr_values_h, sizeof(float) * (uint64_t)nnz,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csr_indptr_d, hA_csrOffsets,sizeof(int) * (uint64_t)(M + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csr_indices_d, hA_columns,sizeof(int) * (uint64_t)nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(A_row_permutation_d, A_row_permutation,sizeof(int) * (uint64_t)M, cudaMemcpyHostToDevice));

    // execute SpMM -- warp up
    for(int iter = 0; iter< warmup_iters;iter++)
    {
        CHECK_CUDA(CudaSpmm(M,K, N, A_nnz,A_row_permutation_d, csr_values_d, csr_indptr_d, csr_indices_d, B_d, C_d, 0));
        CHECK_CUDA(cudaStreamSynchronize(nullptr));
    }

    float elapsed_time = 0;
    cudaEvent_t start_event, stop_event ;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event);
    for(int iter = 0; iter< iters;iter++)
    {
        CHECK_CUDA(CudaSpmm(M,K, N, A_nnz,A_row_permutation_d, csr_values_d, csr_indptr_d, csr_indices_d, B_d, C_d, 0));
        CHECK_CUDA(cudaStreamSynchronize(nullptr));
    }
    cudaEventRecord(stop_event);
    CHECK_CUDA( cudaEventSynchronize(stop_event) )
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event) ;

    CHECK_CUDA(cudaMemcpy(C_h, C_d, sizeof(float) * (uint64_t)M * N, cudaMemcpyDeviceToHost));
    /* device memory deallocation */
    CHECK_CUDA( cudaFree(B_d) )
    CHECK_CUDA( cudaFree(C_d) )
    CHECK_CUDA( cudaFree(csr_values_d) )
    CHECK_CUDA( cudaFree(csr_indptr_d) )
    CHECK_CUDA( cudaFree(csr_indices_d) )
    CHECK_CUDA( cudaFree(A_row_permutation_d) )

    return elapsed_time/iters;

}

/*********************************** helper functions ****************************/
template<class T>
void random_dense(T* mat, int nrows, int ncols, uint32_t max_value)
{
    srand(time(0));
    // srand(0);
    for(int i = 0; i < nrows*ncols; i++)
    { 
        mat[i] = (T)(rand() % max_value);
    }
}

template <class T>
bool match(T* C, T* ans, int m , int n)
{
    T epsilon = 1000;
    bool all_zero = false;
    for(int i = 0; i< m; i++)
    {
        for(int j = 0; j<n; j++)
        {
            if(abs((float)ans[i*n + j] - (float)C[i*n +j]) > (float)epsilon)
            {
                cout<<"!!! test failed at: i="<<i<<" j="<<j<<endl;
                cout<<"ans:"<<(float)ans[i*n + j]<<" != C value:"<<(float)C[i*n + j]<<" diff:"<<(float) ans[i*n + j] - (float)C[i*n + j]<<endl;
                return false;
            }
            // all_zero |=((float)C[i*n +j]==0.);
            // if(j%100000)
            //     cout<<(float)ans[i*n + j]<<" ";
        }
    }
    if(all_zero)
    {
        cout<<"!!! test failed: all result entries are 0!";
        return false;
    }
    return true;
}

int pad(int in, int blksize)
{
    return (in + blksize - 1)/blksize * blksize;
}

void serialMatMul(COMPUTETYPE* A, COMPUTETYPE*B, float* C, int M, int N, int K)
{
    for(int i = 0; i< M; i++)
    {
        for (int j = 0; j< N; j++)
        {
            float sum = 0;
            for( int k = 0; k < K; k++)
            {
                sum+= (float)A[i*K+k] * (float)B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
}

void string_split(string s,string delimiter, vector<string>& tokens)
{
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    token = s.substr(0, pos);
    tokens.push_back(token);
}

void parse_filePath(const string file_path, 
                    string& file_name, 
                    string& reordering_method, int& reordering_blkSize, 
                    int& freezeMatched, float& swapFractionImprove, float& pairwiseFractionImprove, string& reduceAlg, float& affinityFilterMult,
                    bool& isSplit, string& splitType, int&  SplitterTileSize, int&  nnzThreshold,float& splitToOrigNNZRatio)
{
    ///// parse the file name and generate output for csv
    cout<<"\n==================================\n Parsing the file path...\n";
    ////========== get basename
    string basename = file_path.substr(file_path.find_last_of("/") + 1);
    cout<<"basename: "<<basename<<endl;

    ////========== get file_name
    vector<string> basename_splited;
    string_split(basename, ".mtx", basename_splited);  
    // for (string token : basename_splited)
    //     cout<<"token: "<<token<<endl;
    // cout<<endl;
    file_name = basename_splited[0] + ".mtx";
    cout<<"file_name: "<<file_name<<endl;
    if (basename_splited.size()>2) //// Reordered
    {
        cout<<">>>> reordered file\n";
        vector<string> reordering_info_tokens;
        string_split(basename_splited[1], "_", reordering_info_tokens);  
        // for (string token : reordering_info_tokens)
        //     cout<<"token: "<<token<<endl;
        // cout<<endl;
        reordering_method = reordering_info_tokens[1];
        if (basename.find("heurGre") != std::string::npos) 
        {
            reordering_method = "C_Coarsening";

            vector<string> freezeMatched_tokens;
            string_split(reordering_info_tokens[2], ":", freezeMatched_tokens);  
            freezeMatched = stoi(freezeMatched_tokens[1]);
            
            vector<string> swapFractionImprove_tokens;
            string_split(reordering_info_tokens[3], ":", swapFractionImprove_tokens);  
            swapFractionImprove = stof(swapFractionImprove_tokens[1]);
         
            vector<string> pairwiseFractionImprove_tokens;
            string_split(reordering_info_tokens[4], ":", pairwiseFractionImprove_tokens);  
            pairwiseFractionImprove = stof(pairwiseFractionImprove_tokens[1]);

            vector<string> reduceAlg_tokens;
            string_split(reordering_info_tokens[5], ":", reduceAlg_tokens);  
            reduceAlg = reduceAlg_tokens[1];

            vector<string> affinityFilterMult_tokens;
            string_split(reordering_info_tokens[6], ":", affinityFilterMult_tokens);  
            affinityFilterMult = stof(affinityFilterMult_tokens[1]);
        }
        else if (basename.find("NSteps") != std::string::npos) 
        {
            reordering_method = "Py_Coarsening";
            ////========== get reordering_blkSize
            vector<string> nsteps_tokens;
            string_split(reordering_info_tokens[2], "-", nsteps_tokens);  
            // for (string token : nsteps_tokens)
            //     cout<<"token: "<<token<<endl;
            // cout<<endl;

            int step = stoi(nsteps_tokens[1]);
            // cout<<"step:"<<step<<endl;
            reordering_blkSize = (1<<step);
            cout<<"reordering_blkSize:"<<reordering_blkSize<<endl;
        }
        if(basename_splited.size()>3) //// split
        {
            cout<<">>>> splitted file\n";
            isSplit = true;

            vector<string> splitter_info_tokens;
            string_split(basename_splited[2], "_", splitter_info_tokens);  

            splitType = splitter_info_tokens[1];
            cout<<"splitType:"<<splitType<<endl;

            vector<string> SplitterTileSize_tokens;
            string_split(splitter_info_tokens[2], ":", SplitterTileSize_tokens);  

            SplitterTileSize = stoi(SplitterTileSize_tokens[1]);
            cout<<"SplitterTileSize:"<<SplitterTileSize<<endl;

            vector<string> nnzT_tokens;
            string_split(splitter_info_tokens[3], ":", nnzT_tokens);  


            nnzThreshold = stoi(nnzT_tokens[1]);
            cout<<"nnzThreshold:"<<nnzThreshold<<endl;

            vector<string> splitToOrigNNZRatio_tokens;
            string_split(splitter_info_tokens[5], ":", splitToOrigNNZRatio_tokens);  
            // for (string token : nsteps_tokens)
            //     cout<<"token: "<<token<<endl;
            // cout<<endl;

            splitToOrigNNZRatio = stof(splitToOrigNNZRatio_tokens[1]);
            cout<<"splitToOrigNNZRatio:"<<splitToOrigNNZRatio<<endl;
        }
    }
}

void GET_STATS(CSR_Matrix<COMPUTETYPE>* A_CSR, uint32_t blocksize, vector<float>& stats)
{
    unordered_map<uint32_t, unordered_set<uint32_t>> blk_active_rows;
    unordered_map<uint32_t, unordered_set<uint32_t>> blk_active_cols;
    unordered_map<uint32_t, uint32_t> blk_NNZ;
    
    ////// STEP 1      traverse the CSR matrix and collect the stats/info
    for(uint32_t r = 0; r < A_CSR->nrows; r++)
    {
        uint32_t start = A_CSR->rowPtr[r];
        uint32_t end = A_CSR->rowPtr[r+1];
        uint32_t r_blk = r / blocksize;
        for(uint32_t idx = start; idx < end; idx++)
        {
            uint32_t c = A_CSR->cols[idx];
            // COMPUTETYPE val = A_CSR->values[idx];
            uint32_t c_blk = c / blocksize;
            uint32_t blk_offset = r_blk * (A_CSR->ncols / blocksize) + c_blk;

            //// 1. update active rows with in the non empty block
            if(blk_active_rows.find(blk_offset) != blk_active_rows.end()) // if the set corresponding to blk_offset already exists
            {
                blk_active_rows[blk_offset].insert(r);
            }
            else // add the blk_offset
            {
                unordered_set<uint32_t> thisBlockActiveRows;
                thisBlockActiveRows.insert(r);
                blk_active_rows[blk_offset] = thisBlockActiveRows;
            }

            //// 2. update active cols with in the non empty block
            if(blk_active_cols.find(blk_offset) != blk_active_cols.end()) // if blk_offset already exists
            {
                blk_active_cols[blk_offset].insert(c);
            }
            else // add the blk_offset
            {
                unordered_set<uint32_t> thisBlockActiveCols;
                thisBlockActiveCols.insert(c);
                blk_active_cols[blk_offset] = thisBlockActiveCols;
            }

            //// 3. update NNZ with in the non empty block
            if(blk_NNZ.find(blk_offset) != blk_NNZ.end()) // if blk_offset already exists
            {
                blk_NNZ[blk_offset]+= 1;
            }
            else // add the blk_offset
            {
                blk_NNZ[blk_offset] = 1;
            }
        }
    }

    ////// STEP 2      compute the stats
    float ANRS_sum = 0.0;
    float ANCS_sum = 0.0;
    uint32_t total_NNZ = 0;
    float avgInBlkDensity = 0;
    for(auto &nonEmptyBlk : blk_active_rows)
    {
        uint32_t blk_offset = nonEmptyBlk.first;
        unordered_set<uint32_t> &active_rows_set = nonEmptyBlk.second;
        if(blk_active_cols.find(blk_offset) == blk_active_cols.end() || //// each nonEmpyBlk must have a correspoding entry in blk_active_rows, blk_active_cols, and blk_NNZ
           blk_NNZ.find(blk_offset) == blk_NNZ.end())
        {
            assert(false && "Error: improper data/stat gathering!");
        }
        else
        {
            unordered_set<uint32_t> &active_cols_set = blk_active_cols[blk_offset];
            uint32_t _NNZ = blk_NNZ[blk_offset];
            total_NNZ += _NNZ;
            avgInBlkDensity += (_NNZ / (float)(blocksize * blocksize));
            ANRS_sum += ( (float) _NNZ / active_rows_set.size());
            ANCS_sum += ( (float) _NNZ / active_cols_set.size());
        }
    }
    assert(total_NNZ == A_CSR->nnz && "Error: NNZ not match!");
    uint32_t numFillBlks = blk_active_rows.size();
    avgInBlkDensity = avgInBlkDensity / (float) numFillBlks;
    stats[0] = ANRS_sum / (float) numFillBlks;
    stats[1] = ANCS_sum / (float) numFillBlks;
    stats[2] = numFillBlks;
    stats[3] = avgInBlkDensity;
}

float GET_ANXS(CSR_Matrix<COMPUTETYPE>* A_CSR,uint32_t blocksize,bool axis)
{
	unordered_map<uint32_t, unordered_set<uint32_t>> ActiveSegments;
	// unordered_map<uint32_t, uint32_t> bandsNNZ;
    uint32_t nnz = A_CSR->nnz;
	for(uint32_t r = 0; r < A_CSR->nrows; r++)
	{
        uint32_t start = A_CSR->rowPtr[r];
        uint32_t end = A_CSR->rowPtr[r+1];
        uint32_t r_blk = r / blocksize;
        for(uint32_t idx = start; idx < end; idx++)
        {
            uint32_t c = A_CSR->cols[idx];
            // COMPUTETYPE val = A_CSR->values[idx];
            uint32_t c_blk = c / blocksize;
            // uint32_t blk_offset = r_blk * (A_CSR->ncols / blocksize) + c_blk;
            uint32_t band_offset = axis? r_blk:c_blk;
            uint32_t active_Segment = axis? c : r;
            
            //// record the active segmant
            if(ActiveSegments.find(band_offset) != ActiveSegments.end()) // if there exists a set to keep the active segments of this band
                ActiveSegments[band_offset].insert(active_Segment);
            else 
            {
                unordered_set<uint32_t> thisBlockActiveSegments;
                thisBlockActiveSegments.insert(active_Segment);
                ActiveSegments[band_offset] = thisBlockActiveSegments;
            }

            // //// record the nnz
            // if(bandsNNZ.find(band_offset) != bandsNNZ.end()) // if there exists a set to keep the nnz of this band
            //     bandsNNZ[band_offset]++;
            // else 
            // {
            //     bandsNNZ[band_offset]=1;
            // }
	
		}
	}
	//// compute the ANXS
	int sum = 0.0;
	for(auto item : ActiveSegments)
	{
		sum+=(item.second).size();
	}

	// return (float)sum / ActiveSegments.size() ;
    assert(nnz>=sum && "Err: ANXS must be >=1!!");
	return (float)nnz / sum ;
}


/*********************************** main *****************************************/
int main(int argc, char** argv)
{
    cout<<"===================================\n";
    cout<<"Environment variables that you can set:\n\n";
    cout<<"FILE_PATH :: A.mtx file path default:./data/facebook_combined.mtx \n";
    cout<<"TEST :: correctness test mode:0 \n";
    cout<<"BlockSize :: A block size, default:32 \n";
    cout<<"n :: B matrix number of rows, default:64 \n";
    cout<<"iters :: number of iterations, default:5 \n";

    string file_path="./data/facebook_combined.mtx";
    if(std::getenv("FILE_PATH"))
    {
        file_path = std::getenv("FILE_PATH");
    }

    string ComputeType="";
    if(std::getenv("ComputeType"))
    {
        ComputeType = std::getenv("ComputeType");
    }

    string methodsEnv="";
    std::unordered_set<std::string> methods;
    if(std::getenv("methods"))
    {
        methodsEnv = std::getenv("methods");
    }
    // Parse the methods from the environment variable
    std::string methodsString(methodsEnv);
    std::istringstream methodsStream(methodsString);
    std::string method;

    while (std::getline(methodsStream, method, ' ')) {
        methods.insert(method);
    }

    bool test_results = false;
    if(std::getenv("TEST"))
    {
        test_results = (bool) atoi(std::getenv("TEST"));
    }

    int BlockSize = 32;
    if(std::getenv("BlockSize"))
    {
        BlockSize = atoi(std::getenv("BlockSize"));
    }


    uint32_t n = (uint32_t)-1; // if n is not specifically exported then program generates for n in {32,64,...,1024}
    if(std::getenv("n"))
    {
        n = atoi(std::getenv("n"));
    }

    int iters = 1;
    if(std::getenv("iters"))
    {
        iters = atoi(std::getenv("iters"));
    }

    int warmup_iters = 0;
    if(std::getenv("warmup_iters"))
    {
        warmup_iters = atoi(std::getenv("warmup_iters"));
    }

    int get_stats = 0;
    if(std::getenv("get_stats"))
    {
        get_stats = atoi(std::getenv("get_stats"));
    }

    cout<<"\n\nCurrent environment variables values:\n\n";
    cout<<"Current FILE_PATH:"<<file_path<<endl;
    cout<<"Current ComputeType:"<<ComputeType<<endl;
    cout<<"Current TEST:"<<test_results<<endl;
    cout<<"initial BlockSize:"<<BlockSize<<endl;
    cout<<"Current iters:"<<iters<<endl;    
    cout<<"Current warmup_iters:"<<warmup_iters<<endl;    

    vector<int> featureSize;
    if(n == (uint32_t)-1)
    {
        cout<<"n is not specifically exported, the program computes for n in {32,64,...}\n";
        // for(int i : {16,32,64})
        // for(int i : {1,2,4,8,16,32,64,128,256,512,1024,32768, 65536, 131072, 262144, 524288})
        for(int i : {1,2,4,8,16,32,64,128,256,512,1024})
        // for(int i : {16,32,64,128,256,512,1024})
        // for(int i : {1024,2048})
            featureSize.push_back(i);
    }
    else
    {
        cout<<"Current n:"<<n<<endl;
        featureSize.push_back(n);
    }

    ////################# read matrix A from a coo format .mtx + B random generation #######
    SparseCoo<COMPUTETYPE>* A_COO = new SparseCoo<COMPUTETYPE>(file_path); 
    CSR_Matrix<COMPUTETYPE>* A_CSR = new CSR_Matrix<COMPUTETYPE>(*A_COO); // convert A_COO to A_CSR
    A_COO->prepare_for_cusparse();
    uint32_t m = A_COO->dimY;
    uint32_t k = A_COO->dimX;

    vector<vector<float>> STATS;
    if(get_stats)
    {
        for(uint32_t blksize = 2; blksize < 256; blksize*=2)
        {
            float anrs = GET_ANXS(A_CSR, blksize, false);
            float ancs = GET_ANXS(A_CSR, blksize, true);
            STATS.push_back({anrs,ancs});
        }
    }

    preprocess<COMPUTETYPE> * pre = NULL;
    int previous_ELL_dur = 0;
    int previous_ELL_blkSize = 0;

    for(int n : featureSize)
    {
        cout<<"m:"<<m<<" k:"<<k<<" n:"<<n<<endl;
        
        //// Generate Matrix B
        uint64_t B_size = (uint64_t) k*n; 
        COMPUTETYPE* B  = new COMPUTETYPE[B_size];
        random_dense<COMPUTETYPE>(B, k, n, 10);            // random B
        
        float avg_CSR_time = -1;
        float CSR_GFLOPS = -1;
        COMPUTETYPE* CSR_result_host=NULL;
        if(methods.find("cuCSR") != methods.end())
        {
            cout<<"\n################# CSR #######################\n";
            uint64_t CSR_result_size = (uint64_t) m * n;
            
            CSR_result_host = new COMPUTETYPE[CSR_result_size];
            memset (CSR_result_host, 0, sizeof (COMPUTETYPE) * ((uint64_t) m*n));
            
            nvtxEventAttributes_t eventAttrib = {0};
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
            char message[100];
            sprintf(message, "cuCSR_%d", n);
            eventAttrib.message.ascii = message;
            nvtxRangePushEx(&eventAttrib);
            avg_CSR_time =  GPU_CSR_cuSPARSE_NEW(A_CSR->rowPtr, A_CSR->cols, A_CSR->values, A_CSR->nnz, B,CSR_result_host, m, k , n, warmup_iters,iters);
            nvtxRangePop();

            if(avg_CSR_time > 0)
            {
                CSR_GFLOPS = (((double)n * A_CSR->nnz * 2) / avg_CSR_time /1e6);
                cout<<"Cusparse_CSR Avg time(ms): "<<avg_CSR_time<<" GFLOPS:"<<CSR_GFLOPS<<endl;
            }
            else
            {
                cout<<"Err: cuCSR faild!\n";
            }
        }
        float avg_COO_time = -1;
        float COO_GFLOPS = -1;
        if(methods.find("cuCOO") != methods.end())
        {
            cout<<"\n################# COO #######################\n";
            uint64_t COO_result_size = (uint64_t) m * n;
            COMPUTETYPE* COO_result_host = new COMPUTETYPE[COO_result_size];
            memset (COO_result_host, 0, sizeof (COMPUTETYPE) * ((uint64_t) m*n));
            
            nvtxEventAttributes_t eventAttrib = {0};
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
            char message[100];
            sprintf(message, "cuCOO_%d", n);
            eventAttrib.message.ascii = message;
            nvtxRangePushEx(&eventAttrib);

            avg_COO_time = GPU_COO_cuSPARSE_NEW(A_COO->rows, A_COO->cols, A_COO->values, A_COO->nnz, B, COO_result_host, m, k, n, warmup_iters,iters);
            nvtxRangePop();

            if(avg_COO_time > 0)
            {
                COO_GFLOPS = (((double)n * (A_COO->nnz) * 2) / avg_COO_time /1e6);
                cout<<"Cusparse_COO Avg time(ms): "<<avg_COO_time<<" GFLOPS: "<<COO_GFLOPS<<endl;
                if(test_results)
                {
                    cout<<"testing COO  results vs CSR...  \n";
                    if(!match(COO_result_host,CSR_result_host, m, n))
                    {
                        cout<<"COO_result_host and CSR_result_host result does not match at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                        // assert(false && "COO_result_host and CSR_result_host result does not match!!");
                        cout<<"Test failed: COO_result_host and CSR_result_host result does not match!! test failed!"<<endl;
                    }
                    else
                    {
                        cout<<">> Test  passed!! COO_result_host and CSR_result_host result matched!!\n";
                    }
                }
            }
            else
            {
                cout<<"Err: cuCOO faild!\n";
            }
            delete[] COO_result_host;
        }

        vector<float> ELL_times;
        vector<float> ELL_prep_times;
        vector<float> ELL_GFLOPS;
        vector<int> ELL_BlockSize_List;
        if(methods.find("cuELLBlk") != methods.end() && n>=16)
        {
            cout<<"\n################# BlockEll ####################\n";
            if (n==16)
                ELL_BlockSize_List.push_back(16);
            else
                ELL_BlockSize_List.push_back(32);
            for(int blksize : ELL_BlockSize_List)
            {
                uint32_t Ell_m = m;
                uint32_t Ell_k = k;
                uint32_t Ell_n = n;
                cout<<"\n===================================\n";
                BlockSize = blksize;
                cout<<"Current BlockSize:"<<BlockSize<<endl;
                ///// ########### pad dimensions if necessary for ellblk
                if (m%BlockSize)
                {
                    cout<<"Warning: m has to be multiple of BlockSize. Zero padding done to satisfy\n";
                    Ell_m = pad(m, BlockSize);
                    //// pad m dim of A matrix
                    A_COO->dimY = Ell_m;
                    cout<<"new m:"<<Ell_m<<endl;
                }
                if (k%BlockSize)
                {
                    cout<<"Warning: k has to be multiple of BlockSize. Zero padding done to satisfy\n";
                    Ell_k = pad(k, BlockSize);
                    /// pad k dim of A matrix
                    A_COO->dimX = Ell_k;
                    cout<<"new k:"<<Ell_k<<endl;
                }
                if (n%BlockSize)
                {
                    cout<<"Warning: n has to be multiple of BlockSize. Zero padding done to satisfy\n";
                    Ell_n = pad(n, BlockSize);
                    cout<<"new n:"<<Ell_n<<endl;
                }
                COMPUTETYPE* Ell_B = B;
                //// if k or n padded need to pad B
                uint32_t B_padded_k = (Ell_k != k)? Ell_k : k; 
                uint32_t B_padded_n = (Ell_n != n)? Ell_n : n; 
                if (B_padded_k != k || B_padded_n != n)
                {
                    delete[] B;
                    uint64_t B_padded_size = (uint64_t) B_padded_k*B_padded_n; 
                    B  = new COMPUTETYPE[B_padded_size];
                    memset (B, 0, sizeof (COMPUTETYPE) * ((uint64_t) B_padded_size));
                    for(int i = 0; i < k; i++)
                    {
                        for(int j = 0; j < n; j++)
                        {
                            B[i*(B_padded_n)+j] = B[i*n+j];
                        }
                    }
                    Ell_B = B;
                    cout<<"Warning: B matrix zero padded for ELL\n";
                }
                cout<<"Ell_m:"<<Ell_m<<" Ell_k:"<<Ell_k<<" Ell_n:"<<Ell_n<<endl;
                assert((BlockSize <= Ell_m && BlockSize <= Ell_n && BlockSize <= Ell_k) && "Error BlockSize is larger than dimensions! fix the BlockSize to fit!");
                CSR_Matrix<COMPUTETYPE>* A_CSR_padded = new CSR_Matrix<COMPUTETYPE>(*A_COO); // convert A_COO to A_CSR
                auto start_time = std::chrono::high_resolution_clock::now();
                bool performed_preprocess = false;
                if(previous_ELL_blkSize!=BlockSize)
                {
                    pre = new preprocess<COMPUTETYPE> (A_CSR_padded, BlockSize);
                    performed_preprocess = true;
                }
                previous_ELL_blkSize=BlockSize;
                auto end_time = std::chrono::high_resolution_clock::now();
                auto ELL_prep_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                std::cout << "ELL preprocess time: " << ELL_prep_duration.count() << " milliseconds" << std::endl;
                if (performed_preprocess)
                {
                    ELL_prep_times.push_back(ELL_prep_duration.count());
                    previous_ELL_dur = ELL_prep_duration.count();
                }
                else
                    ELL_prep_times.push_back(previous_ELL_dur);
                cout<<"===================================\n";
                cout<<"Blocks info:\n";
                cout<<"Number of fill Blocks:"<<pre->get_num_blocks()<<endl;
                cout<<"\nelements info:\n";
                cout<<"Number of elements:"<<pre->get_num_elems()<<endl;
                if(pre->get_num_elems()!=A_CSR->nnz)
                {
                    cout<<"===================================\n";
                    cout<<"Error At File:"<<__FILE__<<" line:"<<__LINE__<<endl;
                    cout<<"NNZ before generating ELLBlk:"<<(A_CSR->nnz)<<endl;
                    cout<<"NNZ after  generating ELLBlk:"<<pre->get_num_elems()<<endl;
                    assert(pre->get_num_elems()==A_CSR->nnz && "Error: nnz before and after generating ELLBlk does not match.");
                }
                cout<<"===================================\n";
                uint64_t ELL_result_size = (uint32_t) Ell_m * Ell_n;
                float* ELL_result_host = new float[ELL_result_size];
                memset (ELL_result_host, 0, sizeof (float) * ELL_result_size);

                int* ellColIdx = pre->get_ellColIdx();
                COMPUTETYPE*  ellValues = pre->get_ellValues();
                uint32_t A_EllColWidth = pre->get_ellColIdx_width();
                uint32_t A_EllColHeight = pre->get_ellColIdx_height();
                nvtxEventAttributes_t eventAttrib = {0};
                eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
                char message[100];
                sprintf(message, "cuELLBlk-%d-%d",blksize, n);
                eventAttrib.message.ascii = message;
                nvtxRangePushEx(&eventAttrib);
                float avg_gpu_cusparse_EllBlocked = GPU_Compute_With_CuSparse_EllBlocked(ellValues, ellColIdx, A_EllColWidth, A_EllColHeight, BlockSize, Ell_B, ELL_result_host,  Ell_m,  Ell_k ,  Ell_n,  warmup_iters,iters);
                nvtxRangePop();
                if(avg_gpu_cusparse_EllBlocked > 0)
                {
                    float ELL_gflps = (((double)n * A_CSR->nnz * 2) / avg_gpu_cusparse_EllBlocked /1e6);
                    // cout<<"Cusparse_EllBlocked Avg time(ms): "<<avg_gpu_cusparse_EllBlocked<<" GFLOPS:"<<ELL_gflps<<endl;
                    ELL_times.push_back(avg_gpu_cusparse_EllBlocked);
                    ELL_GFLOPS.push_back(ELL_gflps);
                    if(test_results)
                    {
                        // if(!match(ELL_result_host,CSR_result_host, m, n)) /// Testing happen only in original non-padded boundaries
                        // {
                        //     cout<<"Test failed: ELL_result_host and CSR_result_host result does not match!! test failed!"<<endl;
                        //     // assert(false && "Test failed: ELL_result_host and CSR_result_host result does not match");
                        // }
                        // else
                        // {
                        //     cout<<"Test2 passed!! BlockedELL_result_host and CSR_result_host result matched!!\n";
                        // }
                    }
                }
                else
                {
                    ELL_times.push_back(-1);
                    ELL_GFLOPS.push_back(-1);
                }
                delete[] ELL_result_host;
            }
        }

        float avg_gespmm_time = -1;
        float gespmm_GFLOPS = -1;
        if(methods.find("gespmm") != methods.end())
        {
            cout<<"\n################# gespmm #######################\n";
            uint64_t gespmm_result_size = (uint64_t) m * n;
            float* gespmm_result_host = new float[gespmm_result_size];
            memset (gespmm_result_host, 0, sizeof (float) * gespmm_result_size);
            avg_gespmm_time = GPU_gespmm(A_CSR->rowPtr, A_CSR->cols, A_CSR->values, A_CSR->nnz, B, gespmm_result_host, m, k , n,warmup_iters, iters);
            if(avg_gespmm_time > 0)
            {
                gespmm_GFLOPS = (((double)n * A_CSR->nnz * 2) / avg_gespmm_time /1e6);
                cout<<"ge-spmm Avg time(ms): "<<avg_gespmm_time<<" GFLOPS:"<<gespmm_GFLOPS<<endl;
                if(test_results)
                {
                    cout<<"testing gespmm results vs CSR...  \n";
                    // if(!match(gespmm_result_host,CSR_result_host, m, n))
                    // {
                    //     cout<<"gespmm_result_host and CSR_result_host result does not match at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                    //     // assert(false && "gespmm_result_host and CSR_result_host result does not match!!");
                    //     cout<<"Test failed: gespmm_result_host and CSR_result_host result does not match!! test failed!"<<endl;
                    // }
                    // else
                    // {
                    //     cout<<">> Test  passed!! gespmm_result_host and CSR_result_host result matched!!\n";
                    // }
                }
            }
            else
            {
                cout<<"Err: gespmm faild!\n";
            }
            delete[] gespmm_result_host;
        }
        else
        {
            avg_gespmm_time = -1;
            gespmm_GFLOPS = -1;

            /// handle single calls to gespmm algs
            for (const std::string& methodName : methods) 
            {
                if (methodName.find("gespmm") != std::string::npos) 
                {
                    cout<<"\n########### "<<methodName<<" #############\n";
                    uint64_t gespmm_result_size = (uint64_t) m * n;
                    float* gespmm_result_host = new float[gespmm_result_size];
                    memset (gespmm_result_host, 0, sizeof (float) * gespmm_result_size);
                    avg_gespmm_time = GPU_gespmm_singleAlg(A_CSR->rowPtr, A_CSR->cols, A_CSR->values, A_CSR->nnz, B, gespmm_result_host, m, k , n,warmup_iters, iters, methodName);
                    if(avg_gespmm_time > 0)
                    {
                        gespmm_GFLOPS = (((double)n * A_CSR->nnz * 2) / avg_gespmm_time /1e6);
                        cout<<"ge-spmm Avg time(ms): "<<avg_gespmm_time<<" GFLOPS:"<<gespmm_GFLOPS<<endl;
                        if(test_results)
                        {
                            cout<<"testing gespmm results vs CSR...  \n";
                            // if(!match(gespmm_result_host,CSR_result_host, m, n))
                            // {
                            //     cout<<"gespmm_result_host and CSR_result_host result does not match at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                            //     // assert(false && "gespmm_result_host and CSR_result_host result does not match!!");
                            //     cout<<"Test failed: gespmm_result_host and CSR_result_host result does not match!! test failed!"<<endl;
                            // }
                            // else
                            // {
                            //     cout<<">> Test  passed!! gespmm_result_host and CSR_result_host result matched!!\n";
                            // }
                        }
                    }
                    else
                    {
                        cout<<"Err: gespmm faild!\n";
                    }
                    delete[] gespmm_result_host;
                }
            }
        }
        float avg_sputnik_time = -1;
        float sputnik_GFLOPS = -1;
        if(methods.find("sputnik") != methods.end())
        {
            cout<<"\n################# sputnik #######################\n";
            uint64_t sputnik_result_size = (uint64_t) m * n;
            float* sputnik_result_host = new float[sputnik_result_size];
            int* A_row_permutation = new int[m];
            for(int i=0; i<m; i++) 
                A_row_permutation[i] = i;
            memset (sputnik_result_host, 0, sizeof (float) * sputnik_result_size);
            nvtxEventAttributes_t eventAttrib = {0};
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
            char message[100];
            sprintf(message, "sputnik_%d", n);
            eventAttrib.message.ascii = message;
            nvtxRangePushEx(&eventAttrib);
            avg_sputnik_time = GPU_sputnik(A_CSR->rowPtr, A_CSR->cols, A_CSR->values, A_row_permutation, A_CSR->nnz, B, sputnik_result_host, m, k , n, warmup_iters,iters);
            nvtxRangePop();
            if(avg_sputnik_time > 0)
            {
                sputnik_GFLOPS = (((double)n * A_CSR->nnz * 2) / avg_sputnik_time /1e6);
                cout<<"sputnik Avg time(ms): "<<avg_sputnik_time<<" GFLOPS:"<<sputnik_GFLOPS<<endl;
                if(test_results)
                {
                    cout<<"testing sputnik results vs CSR...  \n";
                    // if(!match(sputnik_result_host,CSR_result_host, m, n))
                    // {
                    //     cout<<"sputnik_result_host and CSR_result_host result does not match at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                    //     // assert(false && "sputnik_result_host and CSR_result_host result does not match!!");
                    //     cout<<"Test failed: sputnik_result_host and CSR_result_host result does not match!! test failed!"<<endl;
                    // }
                    // else
                    // {
                    //     cout<<">> Test  passed!! sputnik_result_host and CSR_result_host result matched!!\n";
                    // }
                }
            }
            else
            {
                cout<<"Err: sputnik faild!\n";
            }
            delete[] sputnik_result_host;
            delete[] A_row_permutation;

        }
        if(CSR_result_host!=NULL)
            delete[] CSR_result_host;
        
        //// ########### print all results in csv format
        cout<<"\n######### all results in csv #################\n";
        //// print HEADER
        string headers = "-,filepath,filename,m,k,n,nnz";
        for(auto& method : methods)
        {
            if(method=="cuELLBlk")
                for(int i = 0; i<ELL_BlockSize_List.size();i++)
                {
                    headers += (","+method+std::to_string(ELL_BlockSize_List[i])+"_miliseconds,"+method+std::to_string(ELL_BlockSize_List[i])+"_GFLOPs, ELL_prep_time(ms)");
                }
            else
                headers += (","+method+"_miliseconds,"+method+"_GFLOPs");
        }
        if(get_stats)
        {
            // headers +=",ANRS_16, ANCS_16, numFillBlks_16, avgInBlkDensity_16,ANRS_32, ANCS_32, numFillBlks_32, avgInBlkDensity_32,ANRS_64, ANCS_64, numFillBlks_64, avgInBlkDensity_64,ANRS_128, ANCS_128, numFillBlks_128, avgInBlkDensity_128";
            headers +=",ANRS_2,ANCS_2,ANRS_4,ANCS_4,ANRS_8,ANCS_8,ANRS_16, ANCS_16,ANRS_32, ANCS_32, ANRS_64, ANCS_64, ANRS_128, ANCS_128";
        }
        cout<<headers<<endl;
        string filename = file_path.substr(file_path.find_last_of("/") + 1);

        //// print values
        cout<<"dummy,"<<file_path<<","<<filename<<","<<m<<","<<k<<","<<n<<","<<A_CSR->nnz;

        for(auto& method : methods)
        {
            if(method == "cuCOO")
                cout<<","<<avg_COO_time<<","<<COO_GFLOPS;
            else if(method == "cuCSR")
                cout<<","<<avg_CSR_time<<","<<CSR_GFLOPS;
            // else if(method == "gespmm")
            else if (method.find("gespmm") != std::string::npos)
                cout<<","<<avg_gespmm_time<<","<<gespmm_GFLOPS;
            else if(method == "sputnik")
                cout<<","<<avg_sputnik_time<<","<<sputnik_GFLOPS;
            else if(method == "cuELLBlk") 
            {
                for(int i = 0; i<ELL_GFLOPS.size();i++)
                {
                    cout<<","<<ELL_times[i]<<","<<ELL_GFLOPS[i]<<","<<ELL_prep_times[i];
                }
            }
            else
            {
                cout<<","<<-1<<","<<-1;
                // cout<<"\n\n >>>>>>>>> Err: unknown method:"<<method<<"!"<<endl;
                // assert(false);
            }
        }
        if(get_stats)
        {
            cout<<",";
            for(int i = 0; i < STATS.size(); i++)
            {
                vector<float> &stats = STATS[i];
                for(float stat : stats)
                {
                    cout<<stat<<",";
                }
            }
        }
        cout<<endl;
        delete[] B;
    }
    return 0;
}
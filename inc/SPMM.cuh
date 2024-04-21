#ifndef SPMM_CUH
#define SPMM_CUH

#include "CSR.hpp"
#include <iostream>
#include <mma.h>
#include <cusparse.h>
#include <stdio.h> 
#include <stdlib.h>     
#include <cassert>
#include <cusparseLt.h>       // cusparseLt header
#include <cublas_v2.h>


using namespace std;
using namespace nvcuda;

#define debug_info false
#define test_sparse_tcu false
#ifdef CUBLAS_API_H_
#endif


constexpr int EXIT_UNSUPPORTED = 2;

//Cusp Library

//#include <thrust>
//#include <cusp>

#define TI 32
#define TK 32
#define TJ 32

#define TILE_WIDTH 32

#define TCU_Module_size 16
#define DebugSPMM false


#define WARP_SIZE 32

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GPU with TCU 
// MMA matrix tile dimensions.
#define M_tileDim 16
#define N_tileDim 16
#define K_tileDim 16


// Definition for optimized GEMM with TCU
#define C_LAYOUT wmma::mem_row_major
// Implementation constants.
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)
#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#define SKEW_HALF 16

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at file: %s line %d with error: %s (%d)\n",             \
               __FILE__,__LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at file %s line %d with error: %s (%d)\n",         \
              __FILE__, __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

float sequentioal_spmm(CSR_Matrix *A, double *B, double *C, int m, int k, int n)
{
    /*
        A: pointer to input Sparse matrix MxK
        B: pointer to input Dense matrix  KxN
        C: pointer to output Dense matrix C = AxB  MxN
        returns the time
        Example Sparse matrix A:
                0	1	2	3	4	5	6	7	8
            0	1		2		2			1
            1				3			2		
            2			3						
            3			3		4		4		4
            4	2	2			2		3		1
        CSR representation of A: 
            vals	1,2,2,1,3,2,3,3,4,4,4,2,2,2,3,1
            cols	0,2,4,7,3,6,2,2,4,6,8,0,1,4,6,8
            rptr	0,4,6,7,11,16

        Example Sparse matrix A sorted:
                0	1	2	3	4	5	6	7	8
            0	2	2			2		3		1
            1	1		2		2			1
            2			3		4		4		4
            3				3			2		
            4			3						
        CSR representation of A sorted: 
            vals	2,2,2,3,1,1,2,2,1,3,4,4,4,3,2,3
            cols	0,1,4,6,8,0,2,4,7,2,4,6,8,3,6,2
            rptr	0,5,9,13,15,16
        
        m:5 k:9 n:5
    */

    cudaEvent_t start_event, stop_event;
  

    float elapsed_time_seq = 0;

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event,0);

    for (int row = 0; row < m; row++) // 0..4
    {
        for (int col = 0; col < n; col++) //0..4
        {
            double sum = 0;
            int start = A->rowPtr[row];
            int end = A->rowPtr[row + 1];
            //cout<<row<<","<<col<<","<<start<<","<<end<<endl;
            for (int elemIdx = start; elemIdx < end; elemIdx++)
            {
                //cout<<elemIdx<<"\n";
                sum += A->values[elemIdx] * B[n * A->cols[elemIdx] + col];
            }
            // cout<<row <<","<< col<<"," <<n * row + col<< ","<<sum<<endl;
            C[n * row + col] = sum; // bottle neck
        }
    }
    cudaEventRecord(stop_event,0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_seq, start_event, stop_event);
    return elapsed_time_seq;
}

float GPU_Compute_CuSparseLt(__half* hA, __half* hB, __half* hC , int m, int n, int k,int iters, bool pruned = false)
{
    if(pruned)
        cout<<"\nCuSparseLT\npruned == true, it assumes hA is adready 2:4 and won't prune again\n";
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    auto          order = CUSPARSE_ORDER_ROW;
    auto          opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          type  = CUDA_R_16F;
    auto          compute_type = CUSPARSE_COMPUTE_16F;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 16;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
    auto     A_size         = A_height * lda * sizeof(__half);
    auto     B_size         = B_height * ldb * sizeof(__half);
    auto     C_size         = C_height * ldc * sizeof(__half);
    float alpha = 1.0f;
    float beta  = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    __half *dA, *dB, *dC, *dD, *dA_compressed;
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(d_valid)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    // cout<<"here2\n";
    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;

    // 1. Initialize the library handle: cusparseLtInit().
    CHECK_CUSPARSE( cusparseLtInit(&handle) )

    // 2. Specify the input/output matrix characteristics: 
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )
    // cout<<"here3\n";

    // 3. Initialize the matrix multiplication descriptor and its properties (e.g. operations, compute type, etc.): 
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )
    
    // 4. Initialize the algorithm selection descriptor: 
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    int alg = 0;
    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
                                            &handle, &alg_sel,
                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &alg, sizeof(alg)))
    size_t workspace_size, compressed_size;
    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &alg_sel,
                                                    &workspace_size))
    // cout<<"here4\n";
    // 5. Initialize the matrix multiplication plan: 
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                                workspace_size) )
    //--------------------------------------------------------------------------
    if(!pruned)
    {
        // 6. Prune the A matrix: cusparseLtSpMMAPrune().
        // Prune the A matrix (in-place) and check the correcteness
        CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
            CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    }
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                d_valid, stream) )
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid),
    cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    if (is_valid != 0) 
    {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
        "cusparseLtMatmul will not provide correct results\n");
        return EXIT_FAILURE;
    }
    else
    {
        std::printf("The matrix has been pruned correctly!\n");
    }
    //--------------------------------------------------------------------------
    // 7. Compress the pruned matrix: cusparseLtSpMMACompress().
    // Compress the A matrix
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
        &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA,
    dA_compressed, stream) )
    // cout<<"here5\n";
    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 8. Execute the matrix multiplication: cusparseLtMatmul(). This step can be repeated
    // Perform the matrix multiplication
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    float elapsed_time = 0, elapsed_time_partial = 0 ;
    
    for(int iter = 0; iter< iters;iter++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        
        CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
        &beta, dC, dD, d_workspace, streams, num_streams) )
        
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time_partial, start, stop);
        //cout<<"iteration:"<<iter<<", GPU time for TCU: "<<elapsed_time_partial<<endl;
        elapsed_time += elapsed_time_partial;

    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // device result check
    // matrix A has been pruned
    CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )
    // cout<<"here6\n";

    bool A_std_layout = (is_rowmajor != isA_transposed);
    bool B_std_layout = (is_rowmajor != isB_transposed);

    if(test_sparse_tcu)
    {
        // host computation
        float *hC_result = new float[(uint64_t) m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum  = 0.0f;
                for (int k1 = 0; k1 < k; k1++) {
                    auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                    auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                    sum      += static_cast<float>(hA[posA]) *  // [i][k]
                                static_cast<float>(hB[posB]);   // [k][j]
                }
                auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                hC_result[posC] = sum;  // [i][j]
            }
        }
        // cout<<"here6.1\n";
        // host-device comparison
        int correct = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                auto device_value = static_cast<float>(hC[pos]);
                auto host_value   = hC_result[pos];
                if (device_value != host_value) {
                    // direct floating point comparison is not reliable
                    std::printf("(%d, %d):\t%f vs. %f\n",
                                i, j, host_value, device_value);
                    correct = 0;
                    break;
                }
            }
        }
        delete [] hC_result;
        // cout<<"here6.2\n";
        if (correct)
            std::printf("spmma_example test PASSED\n");
        else
            std::printf("spmma_example test FAILED: wrong result\n");
    }
    //--------------------------------------------------------------------------
    // cout<<"here7\n";
    
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
    // cout<<"here8\n";

    // return EXIT_SUCCESS;
    return  elapsed_time/iters;
}

float GPU_Compute_CuSparseLt(__half* hA, __half* hB, float* hC , int m, int n, int k,int iters, bool pruned = false)
{
    if(pruned)
        cout<<"\nCuSparseLT\npruned == true, it assumes hA is adready 2:4 and won't prune again\n";
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    auto          order = CUSPARSE_ORDER_ROW;
    auto          opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          type  = CUDA_R_16F;
    auto          compute_type = CUSPARSE_COMPUTE_16F;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 16;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
    auto     A_size         = A_height * lda * sizeof(__half);
    auto     B_size         = B_height * ldb * sizeof(__half);
    auto     C_size         = C_height * ldc * sizeof(float);
    float alpha = 1.0f;
    float beta  = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    __half *dA, *dB, *dA_compressed;
    float *dC, *dD;
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(d_valid)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    // cout<<"here2\n";
    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;

    // 1. Initialize the library handle: cusparseLtInit().
    CHECK_CUSPARSE( cusparseLtInit(&handle) )

    // 2. Specify the input/output matrix characteristics: 
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )
    // cout<<"here3\n";

    // 3. Initialize the matrix multiplication descriptor and its properties (e.g. operations, compute type, etc.): 
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )
    
    // 4. Initialize the algorithm selection descriptor: 
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    int alg = 0;
    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
                                            &handle, &alg_sel,
                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &alg, sizeof(alg)))
    size_t workspace_size, compressed_size;
    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &alg_sel,
                                                    &workspace_size))
    // cout<<"here4\n";
    // 5. Initialize the matrix multiplication plan: 
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                                workspace_size) )
    //--------------------------------------------------------------------------
    if(!pruned)
    {
        // 6. Prune the A matrix: cusparseLtSpMMAPrune().
        // Prune the A matrix (in-place) and check the correcteness
        CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
            CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    }
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                d_valid, stream) )
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid),
    cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    if (is_valid != 0) 
    {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
        "cusparseLtMatmul will not provide correct results\n");
        assert(false && "The matrix has been pruned in a wrong way.");
    }
    else
    {
        std::printf("The matrix has been pruned correctly!\n");
    }
    //--------------------------------------------------------------------------
    // 7. Compress the pruned matrix: cusparseLtSpMMACompress().
    // Compress the A matrix
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
        &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA,
    dA_compressed, stream) )
    // cout<<"here5\n";
    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 8. Execute the matrix multiplication: cusparseLtMatmul(). This step can be repeated
    // Perform the matrix multiplication
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    float elapsed_time = 0, elapsed_time_partial = 0 ;
    
    for(int iter = 0; iter< iters;iter++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        
        CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
        &beta, dC, dD, d_workspace, streams, num_streams) )
        
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time_partial, start, stop);
        //cout<<"iteration:"<<iter<<", GPU time for TCU: "<<elapsed_time_partial<<endl;
        elapsed_time += elapsed_time_partial;

    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // device result check
    // matrix A has been pruned
    CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )
    // cout<<"here6\n";

    bool A_std_layout = (is_rowmajor != isA_transposed);
    bool B_std_layout = (is_rowmajor != isB_transposed);

    if(test_sparse_tcu)
    {
        // host computation
        float *hC_result = new float[(uint64_t) m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum  = 0.0f;
                for (int k1 = 0; k1 < k; k1++) {
                    auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                    auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                    sum      += static_cast<float>(hA[posA]) *  // [i][k]
                                static_cast<float>(hB[posB]);   // [k][j]
                }
                auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                hC_result[posC] = sum;  // [i][j]
            }
        }
        // cout<<"here6.1\n";
        // host-device comparison
        int correct = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                auto device_value = static_cast<float>(hC[pos]);
                auto host_value   = hC_result[pos];
                if (device_value != host_value) {
                    // direct floating point comparison is not reliable
                    std::printf("(%d, %d):\t%f vs. %f\n",
                                i, j, host_value, device_value);
                    correct = 0;
                    break;
                }
            }
        }
        delete [] hC_result;
        // cout<<"here6.2\n";
        if (correct)
            std::printf("spmma_example test PASSED\n");
        else
            std::printf("spmma_example test FAILED: wrong result\n");
    }
    //--------------------------------------------------------------------------
    // cout<<"here7\n";
    
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
    // cout<<"here8\n";

    // return EXIT_SUCCESS;
    return  elapsed_time/iters;
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
// Define some error checking macros.

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %s %d\n", stat,_cudaGetErrorEnum(stat), file, line);
   }
}

float GPU_Compute_Cublas(__half* hA, __half* hB, float* hC , int m, int k, int n,int iters)
{
    /*
    From: https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmEx  
    This function is only supported on devices with compute capability 5.0 or later.

    C=αop(A)op(B)+βC

    where α and β are scalars, and A , B and C are matrices stored in >>> column-major format <<<<
    with dimensions op(A) m×k , op(B) k×n and C m×n , respectively. Also, for matrix A

    op(A)={ 
    A	if transa == CUBLAS_OP_N
    AT	if transa == CUBLAS_OP_T
    AH	if transa == CUBLAS_OP_C
    }
    */
    float alpha = 1.0f;
    float beta  = 0.0f;
   
    // create cublas handle 
    cublasHandle_t cublasHandle;
    cublasErrCheck(cublasCreate(&cublasHandle));
  
    // Use tensor cores
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
    //--------------------------------------------------------------------------
    // Device memory management
    auto     A_size         = (uint64_t) m * k * sizeof(__half);
    auto     B_size         = (uint64_t) k * n * sizeof(__half);
    auto     C_size         = (uint64_t) m * n * sizeof(float);
    auto     lda            = k;//(is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = n;//(is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = n;//(is_rowmajor) ? num_C_cols : num_C_rows;
    __half *dA, *dB;
    float *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )    
    //--------------------------------------------------------------------------
    // printf("Running with cuBLAS...\n");
    float elapsed_time = 0, elapsed_time_partial = 0 ;
    
    for(int iter = 0; iter< iters;iter++)
    {
        cudaEvent_t startcublas, stopcublas;
        cudaErrCheck(cudaEventCreate(&startcublas));
        cudaErrCheck(cudaEventCreate(&stopcublas));
        cudaErrCheck(cudaEventRecord(startcublas));
        //col major call: cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,m,n,k,&al,d_a,m,d_b,k,&bet,d_c,m)
        //row major call: cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&al,d_b,n,d_a,k,&bet,d_c,n)
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        n,m,k,
                        &alpha,
                        dB, CUDA_R_16F, ldb,
                        dA, CUDA_R_16F, lda,
                        &beta,
                        dC, CUDA_R_32F, ldc,
                        CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
        cudaErrCheck(cudaEventRecord(stopcublas));
        cudaErrCheck(cudaEventSynchronize(stopcublas));
        cudaErrCheck(cudaEventElapsedTime(&elapsed_time_partial, startcublas, stopcublas))
        elapsed_time += elapsed_time_partial;
    }
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) ) 
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    cublasErrCheck(cublasDestroy(cublasHandle))
    return  elapsed_time/iters;
}
template <class T>
void random_dense_init(T* mat, int nrows, int ncols, uint32_t max_value, int Pattern=0, T fillVal= (T)0)
{
    // generates a matrix of uniformly random values in range [0,max_value]
    // cout<<"==================================\n";
    // cout<<"Generating random dense matrix...\n";
    if(Pattern == 0) // random fill with max_value
    {
        srand(time(0));
        // srand(0);
        for(int i = 0; i < nrows*ncols; i++)
        { 
            mat[i] = rand() % max_value;
        }
    }
    else if(Pattern == 1) // fixed fill with fillVal
    {
        for(int i = 0; i < nrows*ncols; i++)
        { 
            mat[i] = (T)fillVal;
        }
    }
    else if(Pattern == 2) // 1,2,3,4, ...
    {
        for(int i = 0; i < nrows*ncols; i++)
        { 
            mat[i] = (T)i;
        }
    }
    else if(Pattern == 3) // 1, 1, 1, ...;2, 2, 2, ...;... 
    {
        for(int i = 0; i < nrows; i++)
        { 
            for(int j = 0 ; j < ncols; j++)
            {
                mat[i * ncols + j] = (T)i;
            }
        }
    }
    else if(Pattern == 4) // 1, 1, 1, ...;2, 2, 2, ...;... 
    {
        for(int i = 0; i < nrows; i+=(nrows/2))
        { 
            int r = 2*i/nrows; // 0,1
            for(int j = 0 ; j < ncols; j+=(ncols/2))
            {
                int c = 2*j/ncols; //0,1
                for(int ii = 0; ii<nrows/2; ii++)
                    for(int jj = 0; jj<ncols/2; jj++)
                    {
                        int rr = i+ii;
                        int cc = j + jj;
                        mat[rr * ncols + cc] = (T)(2*r+c+1);
                    }
            }
        }
    }
    else if(Pattern == 5)
    {
        for(int i = 0; i < nrows; i++)
        { 
            for(int j = 0 ; j < ncols; j++)
            {
                mat[i * ncols + j] = (T)i;
            }
        }
    } 
}
// Initialize Matrices in Half 

__host__ void init_host_matrices_half(half *a, half *b, float *c, int M_GLOBAL, int N_GLOBAL,  int K_GLOBAL ) {
    for (int i = 0; i < M_GLOBAL; i++) {
      for (int j = 0; j < K_GLOBAL; j++) {
        a[i * K_GLOBAL + j] = (half)(rand() % 3);
      }
    }
  
    for (int i = 0; i < N_GLOBAL; i++) {
      for (int j = 0; j < K_GLOBAL; j++) {
        b[i * K_GLOBAL + j] = (half)(rand() % 3);
      }
    }
  
    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
      c[t] = static_cast<float>(0);
    }
}
  
__host__ void init_host_matrices_float(float *a, float *b, float *c, int M_GLOBAL, int N_GLOBAL,  int K_GLOBAL ) {
    for (int i = 0; i < M_GLOBAL; i++) {
      for (int j = 0; j < K_GLOBAL; j++) {
        a[i * K_GLOBAL + j] = static_cast<float>(rand() % 3);
      }
    }
  
    for (int i = 0; i < N_GLOBAL; i++) {
      for (int j = 0; j < K_GLOBAL; j++) {
        b[i * K_GLOBAL + j] = static_cast<float>(rand() % 3);
      }
    }
  
    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
      c[t] = static_cast<float>(rand() % 3);
    }
}

__global__ void SPMM_basic(int *d_A_rowPtr, int * d_A_cols, double *d_A_values, double *d_B, double *d_C, int m, int k , int n, int nnz)
{
    // ./spmm ~/data/suitesparse_selected_data/pwtk/pwtk.mtx 500 5 => min time 55ms, 11.6Gflops => [211 Glops/s]
    // use shared memory 
    // use loop unrol
    // use tensor cores
    int jj = blockIdx.x,  ii = blockIdx.y;
    int j  = threadIdx.x,  i = threadIdx.y; 

    double sum = 0;
    int row = ii * TI + i;
    int col = jj* TJ + j;
    if(row < m && col < n) 
    {
        int start = d_A_rowPtr[row];
        int end = d_A_rowPtr[row + 1]; 
        for (int elemIdx = start; elemIdx < end; elemIdx++)
        {
            sum += d_A_values[elemIdx] * d_B[n * d_A_cols[elemIdx] + col];
        }
        d_C[n * row + col] = sum; // bottle neck
    }
}
__global__ void SPMM_loop_unrolling(int *d_A_rowPtr, int * d_A_cols, double *d_A_values, double *d_B, double *d_C, int m, int k , int n, int nnz)
{
    // ./spmm ~/data/suitesparse_selected_data/pwtk/pwtk.mtx 500 5 => min time 36ms, 11.6Gflops => [322 Glops/s]

    // use shared memory 
    // use tensor cores
    int jj = blockIdx.x,  ii = blockIdx.y;
    int j  = threadIdx.x,  i = threadIdx.y; 

    double sum = 0;
    double sum1 = 0;
    double sum2 = 0;

    // double sum3 = 0;
    int row = (ii * TI + i);
    int col = 3 * (jj* TJ + j);

    if(row < m && col < n) 
    {
        int start = d_A_rowPtr[row];
        int end = d_A_rowPtr[row + 1]; 

        for (int elemIdx = start; elemIdx < end; elemIdx++)
        {
            double AVal = d_A_values[elemIdx];
            int BRow = d_A_cols[elemIdx];
            int B_Index = n * BRow + col;
            if(B_Index < 0)
                return;
            sum  += AVal * d_B[B_Index];
            if(col+1 < n)
                sum1 += AVal * d_B[B_Index + 1];
            if(col+2 < n)
                sum2 += AVal * d_B[B_Index + 2];
            // if(col+3 < n)
            //     sum3 += AVal * d_B[B_Index + 3];
        }
        int C_index = n * row + col;
        d_C[C_index] = sum; 
        if(col+1 < n)
            d_C[C_index+1] = sum1; 
        if(col+2 < n)
            d_C[C_index+2] = sum2; 
        // if(col+3 < n)
        //     d_C[C_index+3] = sum3; 
    }
}

//Input is in CSR format
// One Warp is calculating one element in output matrix
__global__ void SPMM_shared_warp(int *deviceCSRrow_indx, int *deviceCSRcol_id , double *deviceCSRvalues, double *dmat_in_device , double * dmat_out_device, unsigned int m, unsigned int k , unsigned int n, int nnz)
{   

        __shared__ double vals[TILE_WIDTH+1] ;

        // Global Thread Index
        const int thread_id_x = blockIdx.x  * blockDim.x + threadIdx.x;
        //const int col= blockIdx.x * blockDim.x + threadIdx.x ;
        //Global Warp Index
        const int warp_id = thread_id_x /32 ;
        // Thread Index Within The Warp
        int lane = thread_id_x & (31) ;
        double sum= 0;
        // n here is k in my code

        int irow=warp_id / n ;
        int icol=warp_id & (n-1) ;

        unsigned int numberOfRowCSR = m ;

        // K is #Col of A CSR Matrix
        // unsigned int numberOfRowB = k ;
        unsigned int numberOfColB = n ;

        if( irow < numberOfRowCSR && icol < numberOfColB) {

            int colId ;

            unsigned int row_start = deviceCSRrow_indx[irow];
            unsigned int row_end = deviceCSRrow_indx[irow+1] ;

            //__syncthreads() ;
            vals[threadIdx.x] = 0.0 ;
            //__syncthreads() ;
 
            //__syncwarp();
        #pragma unroll
            for ( unsigned int element = row_start + lane ; element < row_end; element+=32) {
                /* code */
                sum =0 ;

                //colId= A.col_id[i] ;

                colId = deviceCSRcol_id[element] ;
                //printf(" colId = %d thread %d , block %d \n", colId,  icol , irow);

                double value = deviceCSRvalues[element] ;
                double value2 = dmat_in_device[colId * numberOfColB + icol] ;
                sum = value * value2 ;

                //atomicAdd(&vals[threadIdx.x] ,  value * value2 );
                //atomicAdd(&vals[threadIdx.x] ,  sum );
                vals[threadIdx.x] = sum ;
                //printf(" sum =  %d ,thread %d , block %d", sum, icol , irow);
            }
          
            double val=0;
            // //#define FULL_MASK 0xffffffff
            unsigned FULL_MASK = 0xffffffff ;
            // for(int offset = 16 ; offset > 0 ; offset /=2 )
            //     vals += __shfl_down_sync(FULL_MASK, vals, offset);

            unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < 32);

            // unsigned mask = FULL_MASK ;
            if(threadIdx.x < 32)
            {
                val = vals[threadIdx.x];
                for(int offset = 16 ; offset > 0 ; offset /=2 )
                {
                    val += __shfl_down_sync(mask, val, offset);
                }
                vals[threadIdx.x] = val ;

            }
            //__syncwarp();
            //val = val/2;

            if(lane == 0)
            {
                //atomicAdd(&dmat_out_device[irow * numberOfColB + icol] , vals[threadIdx.x]) ;
                   //atomicAdd(&dmat_out_device[irow * numberOfColB + icol] , val) ;
                dmat_out_device[irow * numberOfColB + icol] = vals[threadIdx.x] ;
            }

        }
        
}

//One Warp is calculating multiplication of one row with all column
__global__ void SPMM_shared_warp2(int *deviceCSRrow_indx, int *deviceCSRcol_id , double *deviceCSRvalues, double *dmat_in_device , double * dmat_out_device, unsigned int m, unsigned int k , unsigned int n, int nnz)
{
        __shared__ double vals[TILE_WIDTH+1] ;

        // Global Thread Index
        const int thread_id_x = blockIdx.x  * blockDim.x + threadIdx.x;

        const int warp_id = thread_id_x /32 ;

        int irow=warp_id ;
        int lane = thread_id_x & (31) ;

        unsigned int numberOfRowCSR = m ;

        // K is #Col of A CSR Matrix
        // unsigned int numberOfRowB = k ;
        unsigned int numberOfColB = n ;

        if ( irow < numberOfRowCSR ) {
        #pragma unroll
            for(int icol =0 ; icol < numberOfColB ; icol++)
            {   
                
                int colId;
 
                // int row_start = A.row_indx[iy] ;
                unsigned int row_start = deviceCSRrow_indx[irow];
 
                unsigned int row_end = deviceCSRrow_indx[irow+1] ;
                //printf(" row_end = %d thread %d , block %d \n", row_end,  col , row);
 
                //dmat_out_device[row * K + col] =0;
                //__syncthreads();
                vals[threadIdx.x] = 0 ;
                //__syncthreads();


                for ( int element = row_start + lane ; element < row_end; element+=32) {
                    /* code */

                    //colId= A.col_id[i] ;
                    colId = deviceCSRcol_id[element] ;
                    //printf(" colId = %d thread %d , block %d \n", colId,  col , row);

                    double value = deviceCSRvalues[element] ;
                    double value2 = dmat_in_device[colId * numberOfColB + icol] ;

                    //printf(" colId = %d thread %d , block %d \n", colId,  threadIdx.x , irow);

                    //vals[threadIdx.x] += value + value2 ;
                    // double sum=value * value2;
                    atomicAdd(&vals[threadIdx.x] ,value * value2 );

                    //printf(" sum =  %d ,thread %d , block %d", sum, col , row);
                }

                unsigned FULL_MASK = 0xffffffff ;
                // for(int offset = 16 ; offset > 0 ; offset /=2 )
                //     vals += __shfl_down_sync(FULL_MASK, vals, offset);
    
                unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < 32);
    
                // unsigned mask = FULL_MASK ;

                double val=0;

                if(threadIdx.x < 32)
                {
                    val = vals[threadIdx.x];
                    for(int offset = 16 ; offset > 0 ; offset /=2 )
                    {
                        val += __shfl_down_sync(mask, val, offset);
                    }
                    vals[threadIdx.x] = val ;
    
                }
                if(lane == 0)
                {
                    //atomicAdd(&dmat_out_device[irow * K + icol] , vals[threadIdx.x]) ;
                    dmat_out_device[irow * numberOfColB + icol] = vals[threadIdx.x] ;
                }
            }
        }


}

// Performs an MxNxK GEMM (C=A*B ) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is
// designed for
//       demonstration purposes only to show the CUDA WMMA API use without
//       relying on availability of the shared memory.

// To this TCU Kernel Input should be  Dense MAtrices
// No shared memory
__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld,int n_ld, int k_ld)
{
    //******************** Some Notes :
    // C++ warp matrix operations leverage Tensor Cores to accelerate
    // matrix problems of the form C=A*B.
    // These operations are supported on mixed-precision floating point data 
    // for devices of compute capability 7.0 or higher. This requires co-operation
    // from all threads in a warp. 
    // In addition, these operations are allowed in conditional code only 
    // if the condition evaluates identically across the entire warp, 
    // otherwise the code execution is likely to hang

    // Leading dimensions. Packed with no transpositions.
    // float  alpha=1 ;
    // float beta = 0;

    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid
    int warpSize = 32;
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
     a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
    b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;  
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < k_ld; i += WMMA_K) {
        int aCol = i;
        int aRow = warpM * WMMA_M;

        // N is defined in other version of code as #define N 16 which is # of tiles
        //  #define M_GLOBAL (M * M_TILES) , M_global = M of Matrix A
        //  #define N_GLOBAL (N * N_TILES) , N_global = N of Matrix B
        //  #define K_GLOBAL (K * K_TILES) , K_global = K dimention of Matrix A and B


        int bCol = warpN * N_tileDim;
        int bRow = i;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {

            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication

            // Waits until all warp lanes have arrived at mma_sync, and then performs 
            // the warp-synchronous matrix multiply-accumulate operation D=A*B+C. 
            // The in-place operation, C=A*B+C, is also supported. The value of satf and 
            // template parameters for each matrix fragment must be the same 
            // for all threads in the warp. Also, the template parameters m, n and k must 
            // match between fragments A, B, C and D. This function must be called by all threads in the warp,
            //  or the result is undefined.
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        }

    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha
    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;

    if(cRow < m_ld && cCol < n_ld) {
        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
                               wmma::mem_row_major);
    
        for (int i = 0; i < c_frag.num_elements; i++) {
          //c_frag.x[i] =  acc_frag.x[i] + c_frag.x[i];
          c_frag.x[i] =  acc_frag.x[i] ;
        }
    
        // Store the output
        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
                                wmma::mem_row_major);
    }
}
// }

__global__ void optimized_wmma_gemm(const half *A, const half *B, const float *C,float *D, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL, int M_TILES ,int N_TILES, int K_TILES )
{
    const int M = M_tileDim ;
    const int N = N_tileDim ;
    const int K = K_tileDim ;
    // M =  M_tileDim ;
    // N =  N_tileDim ;
    // K =  K_tileDim ;


    extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

        // This pointer is used to access the C and D matrix tiles this warp computes.
    float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
    (warpId / 2) * SHMEM_STRIDE * K * 2 +
    (warpId % 2) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and
    // from shared memory.
    float *shmem_warp_stream_ptr =
    (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may
    // result in a loss of precision). Zero still needs to be specially handled
    // though.
    //beta /= alpha;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the
    // matrix to the right and down, and selects the next tile to compute. Once
    // there's no such tile, all warps in this CTA exit.
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) 
    {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared
        // memory.
        const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
        #pragma unroll
        for (int i = 0; i < K; i++) 
        {
            typedef int4 copy_t;
            *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) = *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
        }

        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_ROW_TILES; j++) {
            const float *tile_ptr =
                shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

            wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
        }
    }

    __syncthreads();

    // Scale the C matrix.
    // We do not need to scale since in our case it is not D = alpha * A*B + beta* C 

    //     #pragma unroll
    //     for (int i = 0; i < WARP_COL_TILES; i++) {
    // #pragma unroll
    //       for (int j = 0; j < WARP_ROW_TILES; j++) {
    // #pragma unroll
    //         for (int t = 0; t < c[i][j].num_elements; t++) {
    //           c[i][j].x[t] *= beta;
    //         }
    //       }
    //     }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % 4) * 2) : (&B[block_tile_j * N * K_GLOBAL] +N * K_GLOBAL * (warpId % 4) * 2);


    // Go through the global K dimension by a fixed step at a time.
    #pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) 
    {
        // Copy slices of the A and B matrices to shared memory.
        // The first half of the warps in the CTA copy the A matrix, the rest copy
        // the B matrix.
        size_t shmem_idx =
            warpId < (WARPS_PER_BLOCK / 2)
                ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

        // First half of the warp copies the first row / column of the matrix,
        // the second half of the warp copies the next.
        int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
                                (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                        (laneId % CHUNK_COPY_LINE_LANES);

        // Shift the second half of the warp to the next row / column in the
        // shared memory.
        shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

        #pragma unroll
        for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) 
        {
            // Copy 16 bytes at once in each lane.
            *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
                *lane_ptr;
                
            // Advance the global memory pointer and the shared memory index.
            lane_ptr =
                (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
            shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }
        __syncthreads();
        // Compute a grid of C matrix tiles in each warp.
        #pragma unroll
        for (int k_step = 0; k_step < CHUNK_K; k_step++) 
        {
            wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>
                a[WARP_COL_TILES];
            wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>
                b[WARP_ROW_TILES];
            #pragma unroll
            for (int i = 0; i < WARP_COL_TILES; i++) 
            {
                size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];
                wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);
                #pragma unroll
                for (int j = 0; j < WARP_ROW_TILES; j++) 
                {
                    if (i == 0) 
                    {
                        // Load the B matrix fragment once, because it is going to be
                        // reused against the other A matrix fragments.
                        size_t shmem_idx_b = shmem_idx_b_off +
                                            (WARP_ROW_TILES * N) * (warpId % 2) +
                                            (j * N);
                        const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];
                        wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
                    }
                    wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                }
            }
        }
        __syncthreads();
    }

     // Store the D fragments to shared memory.
    #pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) 
    {
        #pragma unroll
        for (int j = 0; j < WARP_ROW_TILES; j++) 
        {
            // #pragma unroll
            // Uniform, point-wise transformations of ALL fragment elements by ALL
            // threads in the warp are well-defined even though element indices
            // within fragment storage are not defined.
            //for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;
            float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
            wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
        }
    }
    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    float *dst_gmem_warp_stream_ptr = &D[gmem_idx];
    #pragma unroll
    for (int i = 0; i < K; i++) {
        *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
            *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }
    __syncthreads();
    }
}

float GPU_Compute(CSR_Matrix *h_A, double *h_B, double *h_C, int m, int k , int n, int iters)
{
    cout<<"\n######################################################################\n";
    cout<<"                     basic kernels...\n";
    cout<<"######################################################################\n";
    // returns the average compute time
    // =================================================
    // Allocate device memory
    int *d_A_rowPtr, *d_A_cols; double *d_A_values;
    double *d_B, *d_C,  *d_Cshared , *d_Cshared_allcol ;

    cudaError_t err = cudaSuccess;
    cudaEvent_t start_event, stop_event;
    cudaEvent_t start_event2, stop_event2;
    cudaEvent_t start_event3, stop_event3;
    cudaEvent_t start_event4, stop_event4;


    cudaMalloc((void**) &d_A_rowPtr, ((m+1)  * sizeof( int ))); // note len(d_A_rowPtr) is m+1
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_A_rowPtr (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_A_cols, (h_A->nnz  * sizeof( int )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_A_cols (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_A_values, (h_A->nnz  * sizeof( double )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_A_values (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_B, ( (k * n)  * sizeof( double )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_C, ((m * n) * sizeof( double )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMalloc((void**) &d_Cshared, ((m * n) * sizeof( double )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMalloc((void**) &d_Cshared_allcol, ((m * n) * sizeof( double )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // =================================================
    // copy input to device
    err = cudaMemcpy(d_A_rowPtr, h_A->rowPtr, ((m+1)  * sizeof( int )), cudaMemcpyHostToDevice);// note len(d_A_rowPtr) is m+1
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A->rowPtr from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_A_cols, h_A->cols, (h_A->nnz  * sizeof( int )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A->cols from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_A_values, h_A->values, (h_A->nnz  * sizeof( double )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A->values from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, ((k * n)  * sizeof( double )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A->values from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // =================================================
    // setup grid/block
    // tile the C matrix. C is m*n, tile it to blocks of TIxTJ
    int gridWidth = (n + TJ - 1 ) / TJ;
    int gridHeight= (m + TI - 1 ) / TI;

    dim3 dimGrid(gridWidth, gridHeight);
    dim3 dimBlock(TJ,TI);

    dim3 dimGrid2(m * n , 1 ,1);
    dim3 dimBlock2(4* TILE_WIDTH,1,1);

    // =================================================
    // lunch the kernel
    float elapsed_time_par1 = 0, partial_elapsed_time_par1 = 0;
    float elapsed_time_par2 = 0, partial_elapsed_time_par2 = 0;
    float elapsed_time_par3 = 0, partial_elapsed_time_par3 = 0;
    float elapsed_time_par4 = 0, partial_elapsed_time_par4 = 0;

    for(int iter = 0; iter< iters;iter++)
    {

        //++++++++++++++++++++++++
        //Basic Kernel

        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event);

        SPMM_basic<<<dimGrid,dimBlock>>>(d_A_rowPtr, d_A_cols,d_A_values, d_B,d_C, m, k , n,h_A->nnz);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch SPMM_basic kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&partial_elapsed_time_par1, start_event, stop_event);

        cout<<"iteration:"<<iter<<", GPU time for SPMM_basic: "<<partial_elapsed_time_par1<<endl;
        elapsed_time_par1 += partial_elapsed_time_par1;

        //=========
        // Loop Unrolling Kernel
        cudaEventCreate(&start_event2);
        cudaEventCreate(&stop_event2);
        cudaEventRecord(start_event2);

        SPMM_loop_unrolling<<<dimGrid,dimBlock>>>(d_A_rowPtr, d_A_cols,d_A_values, d_B,d_C, m, k , n,h_A->nnz);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch SPMM_loop_unrolling kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaEventRecord(stop_event2);
        cudaEventSynchronize(stop_event2);
        cudaEventElapsedTime(&partial_elapsed_time_par2, start_event2, stop_event2);
        cout<<"iteration:"<<iter<<", GPU time for SPMM_loop_unrolling: "<<partial_elapsed_time_par2<<endl;
        elapsed_time_par2 += partial_elapsed_time_par2;


        //+++++++++++++++++++++++
        // Warp Based SpMM with shared memory

        cudaEventCreate(&start_event3);
        cudaEventCreate(&stop_event3);
        cudaEventRecord(start_event3);

        SPMM_shared_warp<<<dimGrid2,dimBlock2>>>(d_A_rowPtr, d_A_cols,d_A_values, d_B, d_Cshared, m, k , n,h_A->nnz);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch SPMM_shared_warp Kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaEventRecord(stop_event3);
        cudaEventSynchronize(stop_event3);
        cudaEventElapsedTime(&partial_elapsed_time_par3, start_event3, stop_event3);
        cout<<"iteration:"<<iter<<", GPU time for SPMM_shared_warp Kernel one warp one row one col: "<<partial_elapsed_time_par3<<endl;
        elapsed_time_par3 += partial_elapsed_time_par3;

        //+++++++++++++++++++++++
        // Warp Based SpMM with shared memory - 2
        // One Warp Do calculation of one row * all column

        cudaEventCreate(&start_event4);
        cudaEventCreate(&stop_event4);
        cudaEventRecord(start_event4);

        SPMM_shared_warp2<<<dimGrid2,dimBlock2>>>(d_A_rowPtr, d_A_cols,d_A_values, d_B, d_Cshared_allcol, m, k , n,h_A->nnz);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch SPMM_shared_warp Kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaEventRecord(stop_event4);
        cudaEventSynchronize(stop_event4);
        cudaEventElapsedTime(&partial_elapsed_time_par4, start_event4, stop_event4);
        cout<<"iteration:"<<iter<<", GPU time for SPMM_shared_warp Kernel one warp one row with all col : "<<partial_elapsed_time_par4<<endl;
        elapsed_time_par4 += partial_elapsed_time_par4;



    }
    // =================================================
    // copy device output to host
    err = cudaMemcpy(h_C, d_C, ((m * n)  * sizeof( double )), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy d_C from Device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // =================================================
    // test the GPU results

    return elapsed_time_par1/iters;

}

float GPU_Compute_With_TCU(half *h_A, half *h_B, float *result_hD, int m, int k , int n, int iters)
{
    cudaError_t err = cudaSuccess;

    int M_GLOBAL = m  ;
    int N_GLOBAL = n  ;
    int K_GLOBAL = k  ;

    // Dummt Matrix which is necessary for WMMA 

    float *C_h = new float[M_GLOBAL * N_GLOBAL];

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        C_h[t] = static_cast<float>(0);
    }

    half *A = NULL;
    half *B = NULL;

    float *C = NULL; // C is 0 matrices since this kernel calculate D= A * B + C
    //Result Matrix in GPU
    float *D = NULL ;

    cudaMalloc(reinterpret_cast<void **>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);

    cudaMalloc(reinterpret_cast<void **>(&C), sizeof(float) * M_GLOBAL * N_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&D), sizeof(float) * M_GLOBAL * N_GLOBAL);
  
    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

    cudaMemcpy(A, h_A, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);

    cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL,cudaMemcpyHostToDevice);

    cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL);

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    // printf("Computing... using simple_wmma_gemm with parameter input\n");

    float elapsed_time = 0, elapsed_time_partial = 0 ;
    
    for(int iter = 0; iter< iters;iter++)
    {
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        //checkCudaErrors(cudaEventRecord(start));

        cudaEventRecord(start);

        //printf("Before simple_wmma \n");
        simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch simple_wmma_gemm kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time_partial, start, stop);
        //cout<<"iteration:"<<iter<<", GPU time for TCU: "<<elapsed_time_partial<<endl;
        elapsed_time += elapsed_time_partial;
    }

    // float *result_hD = NULL;
    // result_hD = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
    cudaMemcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);



    free(C_h);

    cudaFree(reinterpret_cast<void *>(A));
    cudaFree(reinterpret_cast<void *>(B));
    cudaFree(reinterpret_cast<void *>(C));
    cudaFree(reinterpret_cast<void *>(D));

    return  elapsed_time/iters;
}


float GPU_Compute_With_CuSparse(CSR_Matrix *h_A, __half *h_B, float *result_hD, int m, int k , int n, int iters)
{   
    // cout<<"\n######################################################################\n";
    // cout<<"                     CuSparse...\n";
    // cout<<"######################################################################\n";
    // cudaError_t err = cudaSuccess;
    int ldb = n ;
    int ldc = n ;
    int A_nnz = h_A->nnz; 
    __half* values = new __half[A_nnz];
    for(int i = 0; i< A_nnz; i++)
    {
        values[i] = (__half)h_A->values[i];
    }

    // Device memory management
    int *d_A_rowPtr, *d_A_cols ;
    __half *d_A_values , *dB;
    float *dC ;

    CHECK_CUDA( cudaMalloc((void**) &d_A_cols, (h_A->nnz * sizeof(int ))))
    CHECK_CUDA( cudaMalloc((void**) &d_A_values, (h_A->nnz * sizeof(__half))))                  
    CHECK_CUDA( cudaMalloc((void**) &d_A_rowPtr, ((m+1) * sizeof(int)))) 

    CHECK_CUDA( cudaMalloc((void**) &dB, ((k*n) * sizeof(__half)))) 
    CHECK_CUDA( cudaMalloc((void**) &dC, ((m*n) * sizeof(float)))) 

    //Memcpy
    CHECK_CUDA( cudaMemcpy(d_A_rowPtr, h_A->rowPtr , (m+1) * sizeof(int), cudaMemcpyHostToDevice)) 
    CHECK_CUDA( cudaMemcpy(d_A_cols,   h_A->cols,    A_nnz * sizeof(int), cudaMemcpyHostToDevice)) 
    CHECK_CUDA( cudaMemcpy(d_A_values, values,  A_nnz * sizeof(__half), cudaMemcpyHostToDevice)) 
    CHECK_CUDA( cudaMemcpy(dB, h_B,   k*n * sizeof(__half) , cudaMemcpyHostToDevice))

    CHECK_CUDA( cudaMemset(dC, 0, ((m*n) * sizeof(float) ))) 

    

    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    float alpha           = 1.0f;
    float beta            = 0.0f;

    CHECK_CUSPARSE( cusparseCreate(&handle) )

    //Create Sparse Matrix in CSR Format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, m, k, A_nnz,
        d_A_rowPtr, d_A_cols, d_A_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F)) 

    //Create Dense Matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, k, n, ldb, dB, CUDA_R_16F, CUSPARSE_ORDER_ROW) )

    //Create Dense Matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, m, n, ldc, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    float elapsed_time = 0, elapsed_time_partial = 0 ;
    for(int iter = 0; iter< iters;iter++)
    {
        cudaEvent_t start_event, stop_event ;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event) ;

        cudaEventRecord(start_event);

        // Execute SpMM
        CHECK_CUSPARSE( cusparseSpMM(handle,  CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))

        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed_time_partial, start_event, stop_event) ;
        // cout<<"iteration:"<<iter<<", GPU time for cuSparse: "<<elapsed_time_partial<<endl;
        elapsed_time += elapsed_time_partial;
    }
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    // device result check
    CHECK_CUDA( cudaMemcpy(result_hD, dC, ldc* n * sizeof(float),
      cudaMemcpyDeviceToHost) )

    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(d_A_rowPtr) )
    CHECK_CUDA( cudaFree(d_A_values) )
    CHECK_CUDA( cudaFree(d_A_cols) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return elapsed_time / iters ;

}

float GPU_Compute_With_CuSparse_EllBlocked_half(__half *h_A, uint32_t A_ell_blocksize, __half *hB, float *hC, uint32_t m, uint32_t k , uint32_t n, int iters, float alpha=1.0f, float beta=0.0f)
{
    // cout<<"\n######################################################################\n";
    // cout<<"                     CuSparse_EllBlocked...\n";
    // cout<<"######################################################################\n";

    //--------------------------------------------------------------------------
    // Check compute capability
    cudaDeviceProp props;
    CHECK_CUDA( cudaGetDeviceProperties(&props, 0) )
    if (props.major < 7) {
      std::printf("cusparseSpMM with blocked ELL format is supported only "
                  "with compute capability at least 7.0\n");
      return -1;
    }
    //--------------------------------------------------------------------------

    uint32_t   A_num_rows      = m;
    uint32_t   A_num_cols      = k;
    uint32_t   lda             = A_num_cols;
    uint64_t   A_dense_size    = (uint64_t)lda * A_num_rows;
        
    //--------------------------------------------------------------------------
    // 1. calulate blocked_ELL ellColInd
    // NOTE: based on the https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-spmat-create-blockedell
    // ELL-Blocked columns: ellCols = blocksize * max_row_block_number
    // max_row_block_number is maximum number of nonempty blocks in all row blocks
    // The API cusparseDenseToSparse_convert just creates the ellValues and needs the ellCols from the user!! --> look CUDALibrarySamples/cuSPARSE/dense2sparse_blockedell
    //--------------------------------------------------------------------------
    // 1.1 calculate the blocks in each rowblock
    uint32_t numBlocksY = ((A_num_rows + A_ell_blocksize - 1)/A_ell_blocksize);// ceil(nrows/ELL_blocksize)
    vector<unordered_set<uint32_t>>colIdx (numBlocksY, unordered_set<uint32_t>()); // colIdx[row] is a set containing the nonempty block offsets
    
    for(int row = 0; row < A_num_rows; row++)
    {
        for(int col = 0; col<A_num_cols; col++)
        {
            uint32_t blockRowIdx = row / A_ell_blocksize;
            uint32_t blockColIdx = col / A_ell_blocksize;

            if(colIdx[blockRowIdx].find(blockColIdx) == colIdx[blockRowIdx].end()) // if non-empty blockOffset not already added to the relevant set
            {
                colIdx[blockRowIdx].insert(blockColIdx);
            }
        }
    }
    //1.2  ellCols = blocksize * max_row_block_number
    uint32_t max_row_block_number = 0;
    for(auto s : colIdx)
    {
        max_row_block_number = (s.size() > max_row_block_number)? s.size():max_row_block_number;
    }

    uint64_t   A_ell_cols      = (uint64_t) max_row_block_number * A_ell_blocksize;            // ellCols
    uint64_t   A_size          = A_ell_cols * A_num_rows;
    __half*  h_ell_values = new __half[A_size];                              // array to hold BlockedEll ellValue
    
    uint64_t   A_num_blocks    = A_size / ((uint64_t)A_ell_blocksize * A_ell_blocksize);
    uint32_t*  h_ell_columns   = new uint32_t[A_num_blocks];                           // array to hold BlockedEll ellColInd

    //1.3 copy the sets into ELL_columnsIdx
    int i = 0;
    for(auto s : colIdx)
    {
        vector<int> v(s.begin(), s.end());
        sort(v.begin(), v.end());
        for(int elem : v)
        {
            h_ell_columns[i] = elem;
            i++;
        }
    }

    uint32_t   B_num_rows      = k;
    uint32_t   B_num_cols      = n;
    uint32_t   ldb             = B_num_cols;             // B is row-major
    uint64_t   B_size          = (uint64_t)B_num_rows * ldb;
    
    uint32_t   ldc             = B_num_cols;             // C is row-major
    uint64_t   C_size          = (uint64_t)B_num_rows * ldc ;

    //------------------------------------------------------------------
    // 2. Convert Dense to Blocked-ELL format 
        // NOTE: for cusparseDenseToSparse_analysis() the user is resposible to allocate 
        // the memory required by the sparse matrix:
        // Column (ellColInd), value (ellValue) arrays for Blocked-ELL
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_columns;
    __half *dA_values,*dA_dense, *dB;
    float *dC;

    // Device Memory Allocation
    CHECK_CUDA( cudaMalloc((void**) &dA_dense, A_dense_size * sizeof(__half))) // Input Dense
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(int))) // Output ellColInd 
    CHECK_CUDA( cudaMalloc((void**) &dA_values, A_size * sizeof(__half)))     // Output ellValue

    // Copy Data to Device
    CHECK_CUDA( cudaMemcpy(dA_dense, h_A, A_dense_size * sizeof(__half), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, h_ell_columns, A_num_blocks * sizeof(int), cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE Conversion APIs
    cusparseHandle_t     hndl = NULL;
    cusparseSpMatDescr_t matA_SP;
    cusparseDnMatDescr_t matA;
    void*                dBuffer1    = NULL;
    size_t               bufferSize1 = 0;
    CHECK_CUSPARSE( cusparseCreate(&hndl) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, lda, dA_dense,
                                        CUDA_R_16F, CUSPARSE_ORDER_ROW) )

    // Create sparse matrix A_SP in Blocked ELL format
    CHECK_CUSPARSE( cusparseCreateBlockedEll(&matA_SP, A_num_rows, A_num_cols,
                                             A_ell_blocksize, A_ell_cols,
                                             dA_columns, dA_values,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO,
                                             CUDA_R_16F) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        hndl, matA, matA_SP,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize1) )
    
    CHECK_CUDA( cudaMalloc(&dBuffer1, bufferSize1) )


    // execute Dense to Sparse conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(hndl, matA, matA_SP,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer1) )


    // execute Dense to Sparse conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(hndl, matA, matA_SP,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer1) )
    //--------------------------------------------------------------------------
    // // device result check
    // CHECK_CUDA( cudaMemcpy(h_ell_values, dA_values,
    //     A_size * sizeof(__half),
    //     cudaMemcpyDeviceToHost) )
    // CHECK_CUDA( cudaMemcpy(h_ell_columns, dA_columns,
    //         A_num_blocks * sizeof(int),
    //         cudaMemcpyDeviceToHost) )

    // for(int i = 0; i< A_size; i++)
    // {
    //     cout<<(float)h_ell_values[i]<<",";
    // }
    // cout<<"\n^^^^^^^^^^^^\n";
    // for(int i = 0; i< A_num_blocks; i++)
    // {
    //     cout<<(float)h_ell_columns[i]<<",";
    // }

    //--------------------------------------------------------------------------
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) ) //destroy the dense format
    CHECK_CUSPARSE( cusparseDestroy(hndl) )
    CHECK_CUDA( cudaFree(dA_dense) )

    //--------------------------------------------------------------------------
    // 3. perform matmul
    //--------------------------------------------------------------------------

    // Device memory management
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(__half)) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(__half),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),cudaMemcpyHostToDevice) )
    // cout<<"Device memory management completed.\n";
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,CUDA_R_16F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW) )


    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA_SP, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    float elapsed_time = 0, elapsed_time_partial = 0 ;
    for(int iter = 0; iter< iters;iter++)
    {
        cudaEvent_t start_event, stop_event ;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event) ;

        cudaEventRecord(start_event);

        // execute SpMM
        CHECK_CUSPARSE( cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA_SP, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed_time_partial, start_event, stop_event) ;
        // cout<<"iteration:"<<iter<<", GPU time for ELLBlocked: "<<elapsed_time_partial<<endl;
        elapsed_time += elapsed_time_partial;
    }
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),cudaMemcpyDeviceToHost) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA_SP) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device memory deallocation

    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) ) 
    return elapsed_time/iters ;
}

float GPU_Compute_With_CuSparse_EllBlocked_half(__half *hA_values, int* hA_columns, int A_EllColWidth,int A_EllColHeight, uint32_t A_ell_blocksize, __half *hB, float *hC, uint32_t m, uint32_t k , uint32_t n, int iters, float alpha=1.0f, float beta=0.0f)
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
    uint64_t   B_size          = (uint64_t)B_num_rows * ldb;
    
    uint32_t   C_num_rows      = A_num_rows;
    uint32_t   C_num_cols      = B_num_cols;
    uint32_t   ldc             = C_num_cols;             // C is row-major
    uint64_t   C_size          = (uint64_t)C_num_rows * ldc ;

    int    *dA_columns;
    __half *dA_values, *dB;
    float  *dC;
    uint64_t A_num_blocks  = A_EllColWidth * A_EllColHeight;
    uint64_t A_Values_size = A_num_blocks  * A_ell_blocksize * A_ell_blocksize;
    uint64_t A_ell_cols    = (uint64_t) A_EllColWidth * A_ell_blocksize;            // ellCols

    // Device Memory Allocation
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(int)))       // Output ellColInd 
    CHECK_CUDA( cudaMalloc((void**) &dA_values, A_Values_size * sizeof(__half)))    // Output ellValue

    // Copy Data to Device
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_num_blocks * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_Values_size * sizeof(__half), cudaMemcpyHostToDevice) )
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
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
    // Device memory management
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(__half)) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(__half),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),cudaMemcpyHostToDevice) )

    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB,CUDA_R_16F, CUSPARSE_ORDER_ROW) )
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
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    float elapsed_time = 0, elapsed_time_partial = 0 ;
    for(int iter = 0; iter< iters;iter++)
    {
        cudaEvent_t start_event, stop_event ;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event) ;

        cudaEventRecord(start_event);

        // execute SpMM
        CHECK_CUSPARSE( cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA_SP, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed_time_partial, start_event, stop_event) ;
        elapsed_time += elapsed_time_partial;
    }
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


float GPU_Compute_With_Optimized_TCU(half *h_A, half *h_B, float *result_hD, int m, int k , int n)
{
    int M_GLOBAL = m  ;
    int N_GLOBAL = n  ;
    int K_GLOBAL = k  ;


    float *C_h = NULL;
    C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        C_h[t] = static_cast<float>(0);
    }

    half *A = NULL;
    half *B = NULL;

    float *C = NULL; // C is 0 matrices since this kernel calculate D= A * B + C
    //Result Matrix in GPU
    float *D = NULL ;

    cudaMalloc(reinterpret_cast<void **>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&B),sizeof(half) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&C), sizeof(float) * M_GLOBAL * N_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&D), sizeof(float) * M_GLOBAL * N_GLOBAL);
  
    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

    cudaMemcpy(A, h_A, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);

    cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL,cudaMemcpyHostToDevice);

    cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL);

    printf("============ Computing... using optimized_wmma_gemm with parameter input =======\n");

    enum {
        // Compute the right amount of shared memory to request.
        // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
        // per-CTA chunks
        // of the A and B matrices. Therefore, the right amount to request is the
        // maximum of those
        // two numbers.
        SHMEM_SZ = MAX(sizeof(half) * (BLOCK_COL_TILES * M_tileDim) * (CHUNK_K * K_tileDim + SKEW_HALF) * 2,
            M_tileDim * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N_tileDim * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
    };


    printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);
    printf("Device available shared memory size: %d KB\n", 164 );

    cudaFuncSetAttribute(optimized_wmma_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //on Ampere100 GPU at CHPC DeviceProf.multiProcessoeCount = 108
    cudaEventRecord(start);
    optimized_wmma_gemm<<<108 , THREADS_PER_BLOCK, SHMEM_SZ>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, m, n, k);
    cudaEventRecord(stop) ;
    cudaEventSynchronize(stop) ;

    cudaMemcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost) ;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time of GPU_Compute With Optimzied : %f ms\n", milliseconds);


    free(C_h);

    cudaFree(reinterpret_cast<void *>(A));
    cudaFree(reinterpret_cast<void *>(B));
    cudaFree(reinterpret_cast<void *>(C));
    cudaFree(reinterpret_cast<void *>(D));

    return milliseconds;

}

float GPU_Compute_With_Optimized_TCU_Random_Matrices(int m, int k , int n)
{
    int M_GLOBAL = m * M_tileDim ;
    int N_GLOBAL = n * N_tileDim ;
    int K_GLOBAL = k * K_tileDim ;

    printf("M: %d (%d x %d)\n", M_GLOBAL, m, M_tileDim);
    printf("K: %d (%d x %d)\n", K_GLOBAL, k, K_tileDim);
    printf("N: %d (%d x %d)\n", N_GLOBAL, n, N_tileDim);


    half *A_h = NULL;
    half *B_h = NULL;
    float *C_h = NULL;

    A_h = (half *)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h = (half *)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

    half *A = NULL;
    half *B = NULL;

    float *C = NULL; // C is 0 matrices since this kernel calculate D= A * B + C
    //Result Matrix in GPU
    float *D = NULL ;

    float *result_hD = NULL;
    result_hD = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

    cudaMalloc(reinterpret_cast<void **>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&C), sizeof(float) * M_GLOBAL * N_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&D), sizeof(float) * M_GLOBAL * N_GLOBAL);
   
    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

    init_host_matrices_half(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    //init_host_matrices_half(A_h, B_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    printf("Preparing data for Optimized GPU Kernel with TCU ...\n");
    cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL,cudaMemcpyHostToDevice);

    //cudaMemset(C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL);
    cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL);

    enum {
        // Compute the right amount of shared memory to request.
        // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
        // per-CTA chunks
        // of the A and B matrices. Therefore, the right amount to request is the
        // maximum of those
        // two numbers.
        SHMEM_SZ = MAX(sizeof(half) * (BLOCK_COL_TILES * M_tileDim) * (CHUNK_K * K_tileDim + SKEW_HALF) * 2,
            M_tileDim * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N_tileDim * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
    };


    printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);
    printf("Device available shared memory size: %d KB\n", 164 );

    cudaFuncSetAttribute(optimized_wmma_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //on Ampere100 GPU at CHPC DeviceProf.multiProcessoeCount = 108
    cudaEventRecord(start);
    optimized_wmma_gemm<<<108 , THREADS_PER_BLOCK, SHMEM_SZ>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, m, n, k);
    cudaEventRecord(stop) ;
    cudaEventSynchronize(stop) ;

    cudaMemcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost) ;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time of GPU_Compute_With_Optimized_TCU_Random_Matrices: %f ms\n", milliseconds);

    free(A_h);
    free(B_h);
    free(C_h);
    free(result_hD);

    cudaFree(reinterpret_cast<void *>(A));
    cudaFree(reinterpret_cast<void *>(B));
    cudaFree(reinterpret_cast<void *>(C));
    cudaFree(reinterpret_cast<void *>(D));

    return milliseconds;

}

// m , n , k are # TILES for M , N , K dimension 
// Final Dimension of A and B are M_Global, N_Global and K_Global
// It creates some random A and B matrices based on m, k ,n

float GPU_Compute_With_TCU_Random_Matrices(int m, int k , int n)
{
    int M_GLOBAL = m * M_tileDim ;
    int N_GLOBAL = n * N_tileDim ;
    int K_GLOBAL = k * K_tileDim ;

    printf("M: %d (%d x %d)\n", M_GLOBAL, m, M_tileDim);
    printf("K: %d (%d x %d)\n", K_GLOBAL, k, K_tileDim);
    printf("N: %d (%d x %d)\n", N_GLOBAL, n, N_tileDim);
    
    half *A_h = NULL;
    half *B_h = NULL;
    float *C_h = NULL;

    A_h = (half *)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h = (half *)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

    half *A = NULL;
    half *B = NULL;

    float *C = NULL; // C is 0 matrices since this kernel calculate D= A * B + C
    //Result Matrix in GPU
    float *D = NULL ;

    cudaMalloc(reinterpret_cast<void **>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&C), sizeof(float) * M_GLOBAL * N_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&D), sizeof(float) * M_GLOBAL * N_GLOBAL);
   
    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

    init_host_matrices_half(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    //init_host_matrices_half(A_h, B_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    printf("Preparing data for GPU...\n");

    cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL,cudaMemcpyHostToDevice);

    //cudaMemset(C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL);
    cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL);

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Computing... using simple_wmma_gemm kernel \n");

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //checkCudaErrors(cudaEventRecord(start));

    cudaEventRecord(start);


    simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL,K_GLOBAL);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms for TCU GeMM WMMA\n", milliseconds);

    float *result_hD = NULL;
    result_hD = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

    cudaMemcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);

    free(A_h);
    free(B_h);
    free(C_h);

    cudaFree(reinterpret_cast<void *>(A));
    cudaFree(reinterpret_cast<void *>(B));
    cudaFree(reinterpret_cast<void *>(C));
    cudaFree(reinterpret_cast<void *>(D));

    return milliseconds;
}

void GPU_Compute_Multi_Kernel(CSR_Matrix *h_A, half *h_B, float *h_C, int m, int k , int n, int iters, int A_TI, int Min_NNZ_to_use_TCU, int nnz_moving_avg_window_size,float nnz_moving_avg_threashold, float TCU_FIX_NNZ, float cut_margin)
{
    cout<<"\n######################################################################\n";
    cout<<"                     Hybrid Kernel Computation...\n";
    cout<<"######################################################################\n";

    //###########################################################################################
    //    sort the sparse matrix rows based on the row NNZ
    //###########################################################################################
    h_A->sort_rows();
    //h_A->print_topk_sorted(20);

    //###########################################################################################
    //    cut computation approach 1: using the Moving Average
    //###########################################################################################
    int mAvgCut_height = h_A->cut(Min_NNZ_to_use_TCU, nnz_moving_avg_window_size, nnz_moving_avg_threashold, cut_margin);
    cout<<"cut_margin:"<<cut_margin<<endl;
    cout<<"Cut height before margin:"<<mAvgCut_height<<endl;

    cout<<"Moving average cut height:"<<mAvgCut_height<<endl;
    cout<<"==================================\n";

    int Cut_height = mAvgCut_height;

    int TCU_TILE_ROWs = Cut_height;


    cout<<"##################################\n";
    cout<<"          TCU Kernel\n";
    cout<<"##################################\n";

    if(TCU_TILE_ROWs % TCU_Module_size != 0) //e.g. if rows is not divisible by 16 then pad it to be divisible by 16
    {
        int floor_tcu_blocks = TCU_TILE_ROWs/TCU_Module_size;
        TCU_TILE_ROWs = (floor_tcu_blocks + 1) * TCU_Module_size < h_A->nrows ? (floor_tcu_blocks + 1) * TCU_Module_size: floor_tcu_blocks * TCU_Module_size;
    }

    int TCU_TILE_COLs = h_A->ncols;
    if(TCU_TILE_COLs % TCU_Module_size != 0) //e.g. if cols is not divisible by 16 then pad it to be divisible by 16
    {
        TCU_TILE_COLs = (TCU_TILE_COLs/TCU_Module_size + 1) * TCU_Module_size;
    }
    
    //double *h_A_TCU_Partition = new double[TCU_TILE_ROWs * TCU_TILE_COLs];
    half *h_A_TCU_Partition = NULL ;
    h_A_TCU_Partition = (half*)malloc(sizeof(half) * TCU_TILE_ROWs * TCU_TILE_COLs );
    //new half[TCU_TILE_ROWs * TCU_TILE_COLs];
    //cout<<"Zero FIll is starting"<<endl;

    fill_n(h_A_TCU_Partition, TCU_TILE_ROWs * TCU_TILE_COLs, 0.0f);// zero fill this can be time-comsuming !!!!!!!!
    //cout<<"Zero FIll is Done"<<endl;

    h_A->loadSortedToDense(h_A_TCU_Partition, Cut_height, TCU_TILE_ROWs, TCU_TILE_COLs);
    //cout<<"load Sorted to Dense is done"<<endl;
    
    //===========================
    // TODO: TCU_API(h_A_TCU_Partition, h_B, h_C)
    //===========================
    float tcu_time = 0;
    
    half *B_h = NULL;
    B_h = (half *)malloc(sizeof(half) * TCU_TILE_COLs * n);

    for(int i = 0; i < TCU_TILE_COLs; i++) { 
        for(int j = 0 ; j < n; j++){
            B_h[i * n + j] =(half) (rand()/float(RAND_MAX) * 10); 
        }
    } 

    float *D_h = NULL;
    D_h = (float *)malloc(sizeof(float) * TCU_TILE_ROWs * n);

    cout<< " TCU Multi GPU" << endl ;

    cout<< " Dense Partition of Sparse Input Row : "<<TCU_TILE_ROWs <<endl;
    cout<< " Dense Partition of Sparse Input Column : "<<TCU_TILE_COLs << endl;
    cout<< " n is : "<<n << endl;

    tcu_time=GPU_Compute_With_TCU(h_A_TCU_Partition, B_h, D_h , TCU_TILE_ROWs, TCU_TILE_COLs , n, iters);

    cout<<"tcu_time in GPU_Compute_Multi_Kernel = "<<tcu_time<<endl;
    
    cout<<"Compute is done for dense partition"<<endl;


    cout<<"##################################\n";
    cout<<"        Normal  Kernel\n";
    cout<<"##################################\n";
    
    //  load the sparse partition into a CSR partition to feed to normal kernel
    CSR_Matrix* h_A_normal_partition  = new CSR_Matrix();
    h_A->loadSortedToCSR(h_A_normal_partition,Cut_height);

    // h_A_normal_partition->print_topk(20);

    cout<<"Running the sparse partition of  Multi kernel...\n";

    // Allocate device memory
    m = h_A_normal_partition->nrows;
    int *d_A_rowPtr, *d_A_cols; double *d_A_values;
    double *d_B, *d_C;

    cudaError_t err = cudaSuccess;
    cout<<"m:"<<h_A_normal_partition->nrows<<" k:"<<h_A_normal_partition->ncols<<" n:"<<n<<" nnz:"<<h_A_normal_partition->nnz<<endl;

    cudaMalloc((void**) &d_A_rowPtr, ((m+1)  * sizeof( int ))); // note len(d_A_rowPtr) is m+1
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_A_rowPtr (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_A_cols, (h_A_normal_partition->nnz  * sizeof( int )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_A_cols (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_A_values, (h_A_normal_partition->nnz  * sizeof( double )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_A_values (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_B, ( (k * n)  * sizeof( double )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_C, ((m * n) * sizeof( double )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // copy input to device
    err = cudaMemcpy(d_A_rowPtr, h_A_normal_partition->rowPtr, ((m+1)  * sizeof( int )), cudaMemcpyHostToDevice);// note len(d_A_rowPtr) is m+1
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A_normal_partition->rowPtr from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_A_cols, h_A_normal_partition->cols, (h_A_normal_partition->nnz  * sizeof( int )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A_normal_partition->cols from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_A_values, h_A_normal_partition->values, (h_A_normal_partition->nnz  * sizeof( double )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A_normal_partition->values from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, ((k * n)  * sizeof( double )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A->values from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float elapsed_time = 0, elapsed_time_partial = 0 ;
    cudaEvent_t start_event, stop_event;
   for(int iter = 0; iter< iters;iter++)
    {
        // lunch the kernel
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event);
        int gridWidth = (n + TJ - 1 ) / TJ;
        int gridHeight= (m + TI - 1 ) / TI;
        dim3 dimGrid(gridWidth, gridHeight);
        dim3 dimBlock(TJ,TI);
        SPMM_loop_unrolling<<<dimGrid,dimBlock>>>(d_A_rowPtr, d_A_cols,d_A_values, d_B,d_C, m, k , n,h_A_normal_partition->nnz);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch SPMM_loop_unrolling kernel (error code %s)!\n", cudaGetErrorString(err));
            fprintf(stderr, "iteration:%d\ngridWidth:%d gridHeight:%d \nTJ:%d TI:%d\nm:%d k:%d n:%d nnz:%d\n", iter, gridWidth, gridHeight, TJ, TI, m, k, n, h_A_normal_partition->nnz);
            exit(EXIT_FAILURE);
        }
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed_time_partial, start_event, stop_event);
        cout<<"iteration:"<<iter<<", GPU time for sparse Kernel: "<<elapsed_time_partial<<endl;
        elapsed_time += elapsed_time_partial;
    }
    cout<<"GPU AVG time for sparse partition of GPU Multi kernel: "<<elapsed_time/iters<<endl;
   
    cout<<"\n##################################\n";
    cout<<"Total Hybrid Kernel time:"<<tcu_time + elapsed_time/iters<<endl;
    cout<<"##################################\n";
    
    // free memory
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_values);
    cudaFree(d_A_cols);

    free(D_h) ;
    free(B_h) ;

}

void GPU_Compute_Multi_Kernel_Opt(CSR_Matrix *h_A, half *h_B, float *h_C, int m, int k , int n, int iters, int A_TI, int Min_NNZ_to_use_TCU, int nnz_moving_avg_window_size,float nnz_moving_avg_threashold, float TCU_FIX_NNZ)
{
    h_A->sort_rows();
    cout<<"sorted matrix:\n";

     //================= basic cut
     int basic_cut_height = h_A->cut_basic(TCU_FIX_NNZ);
     cout<<"basic cut height:"<<basic_cut_height<<endl;
 
     int TCU_TILE_ROWs = basic_cut_height;
     if(TCU_TILE_ROWs % TCU_Module_size != 0) //e.g. if rows is not divisible by 16 then pad it to be divisible by 16
     {
         TCU_TILE_ROWs = (TCU_TILE_ROWs/TCU_Module_size + 1) * TCU_Module_size;
     }
 
     int TCU_TILE_COLs = h_A->ncols;
     if(TCU_TILE_COLs % TCU_Module_size != 0) //e.g. if cols is not divisible by 16 then pad it to be divisible by 16
     {
         TCU_TILE_COLs = (TCU_TILE_COLs/TCU_Module_size + 1) * TCU_Module_size;
     }
    
     half *h_A_TCU_Partition = new half[TCU_TILE_ROWs * TCU_TILE_COLs];

     fill_n(h_A_TCU_Partition, TCU_TILE_ROWs * TCU_TILE_COLs, 0.0f);
     h_A->loadSortedToDense(h_A_TCU_Partition, basic_cut_height, TCU_TILE_ROWs, TCU_TILE_COLs);

     float tcu_time = 0;
    
     half *B_h = NULL;
     B_h = (half *)malloc(sizeof(half) * TCU_TILE_COLs * n);
 
     for(int i = 0; i < TCU_TILE_COLs; i++) { 
         for(int j = 0 ; j < n; j++){
             B_h[i * n + j] =(half) (rand()/double(RAND_MAX) * 10); 
         }
     } 

    float *C_h = NULL;
    C_h = (float *)malloc(sizeof(float) * TCU_TILE_ROWs * n);

    cout<< " TCU Multi GPU" << endl ;
    cout<< " Dense Partition of Sparse Input Row : "<<TCU_TILE_ROWs <<endl;
    cout<< " Dense Partition of Sparse Input Column : "<<TCU_TILE_COLs << endl;

    tcu_time=GPU_Compute_With_Optimized_TCU(h_A_TCU_Partition, h_B, C_h , TCU_TILE_ROWs, TCU_TILE_COLs , n);

    cout<<"tcu_time in Optimized GPU_Compute_Multi_Kernel = "<<tcu_time<<endl;
    
    cout<<"Compute is done for dense partition"<<endl;

      //  Rest of Row SpMM
    
    //===========================
    //
    //
    //  load the sparse partition into a CSR partition to feed to normal kernel
    //===========================

    CSR_Matrix* h_A_normal_partition  = new CSR_Matrix();
    h_A->loadSortedToCSR(h_A_normal_partition,basic_cut_height);
    // h_A_normal_partition->print_topk(20);

    // //===========================   
    // // normal kernel
    // //===========================
    cout<<"running the normal spmm kernel...\n";
    cout<<"==================================\n";
    // returns the average compute time
    // =================================================
    // Allocate device memory
    m = h_A_normal_partition->nrows;
    int *d_A_rowPtr, *d_A_cols; double *d_A_values;
    double *d_B, *d_C;

    cudaError_t err = cudaSuccess;
    cudaEvent_t start_event2, stop_event2;
    float partial_elapsed_time_par2 = 0;
    // cudaEvent_t start_event3, stop_event3;
    // cudaEvent_t start_event4, stop_event4;
    cout<<"m:"<<h_A_normal_partition->nrows<<" k:"<<h_A_normal_partition->ncols<<" n:"<<n<<" nnz:"<<h_A_normal_partition->nnz<<endl;

    cudaMalloc((void**) &d_A_rowPtr, ((m+1)  * sizeof( int ))); // note len(d_A_rowPtr) is m+1
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_A_rowPtr (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_A_cols, (h_A_normal_partition->nnz  * sizeof( int )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_A_cols (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_A_values, (h_A_normal_partition->nnz  * sizeof( double )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_A_values (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_B, ( (k * n)  * sizeof( double )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**) &d_C, ((m * n) * sizeof( double )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

     // =================================================
    // copy input to device
    err = cudaMemcpy(d_A_rowPtr, h_A_normal_partition->rowPtr, ((m+1)  * sizeof( int )), cudaMemcpyHostToDevice);// note len(d_A_rowPtr) is m+1
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A_normal_partition->rowPtr from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_A_cols, h_A_normal_partition->cols, (h_A_normal_partition->nnz  * sizeof( int )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A_normal_partition->cols from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_A_values, h_A_normal_partition->values, (h_A_normal_partition->nnz  * sizeof( double )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A_normal_partition->values from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, ((k * n)  * sizeof( double )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_A->values from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaEventCreate(&start_event2);
    cudaEventCreate(&stop_event2);
    cudaEventRecord(start_event2);
    int gridWidth = (n + TJ - 1 ) / TJ;
    int gridHeight= (m + TI - 1 ) / TI;
    dim3 dimGrid(gridWidth, gridHeight);
    dim3 dimBlock(TJ,TI);
    SPMM_loop_unrolling<<<dimGrid,dimBlock>>>(d_A_rowPtr, d_A_cols,d_A_values, d_B,d_C, m, k , n,h_A->nnz);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch SPMM_loop_unrolling kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaEventRecord(stop_event2);
    cudaEventSynchronize(stop_event2);
    cudaEventElapsedTime(&partial_elapsed_time_par2, start_event2, stop_event2);
    cout<<"GPU time for sparse partition of GPU Multi kernel: "<<partial_elapsed_time_par2<<endl;

    cudaFree(d_B);
    cudaFree(d_C);

    free(C_h) ;
    free(B_h) ;
}

void GPU_Compute_Multi_Kernel_2D(SparseCoo& h_A_COO, double *h_B, double *h_C, int m, int k , int n, int iters, int A_TI, int A_TK, int THRESHOLD)
{
    h_A_COO.tile_2D(A_TI, A_TK);
    for(auto& it : h_A_COO.tiles)
    {
        int tile_row = it.first / TK;
        int tile_col = it.first % TK;
        if(it.second.size() > THRESHOLD) //choose the TCU dense kernel
        {
            // define dense tile and initialize it to 0
            int size = A_TI*A_TK;
            double* A_dense_tile = new double[size];
            fill_n(A_dense_tile, size, 0);// >> this can be time-comsuming !!!!!!!!

            // load elements to the dense tile
            for(int i : it.second)
            {
                int elem_row_in_tile = h_A_COO.data_vector[i].row - tile_row*TI; // element row index in its tile
                int elem_col_in_tile = h_A_COO.data_vector[i].col - tile_col*TK; // element col index in its tile
                int elem_val = h_A_COO.data_vector[i].val; 
                A_dense_tile[elem_row_in_tile*TK + elem_col_in_tile] = elem_val;
            }
            // ==================================
            // call the TCU_dense_API(A_dense_tile) kernel
            // ==================================
        }
        else // choose the normal spmm kernel
        {
            // ==================================
            // load elements to the COO tile
            // convert COO data into CSR
            // call the normal spmm
            // ==================================
        }
    }
}
#endif

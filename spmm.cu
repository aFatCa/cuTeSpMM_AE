#include "kernel.cu"
#include <cusparse.h>
#ifdef A100
#define MAX_SM_SIZE 163840
#else
#define MAX_SM_SIZE 94208
#endif
#ifndef TN
#define TN 2
#endif
#if TN == 4
#define WARPS_PER_BLOCK_16 1
#define THREADS_PER_BLOCK_16 32
#define ILP_LEVEL_16 2
#define GRID_DIM_Y_16 1

#define WARPS_PER_BLOCK_32 1
#define THREADS_PER_BLOCK_32 32
#define ILP_LEVEL_32 4
#define GRID_DIM_Y_32 1

#define WARPS_PER_BLOCK_64 2
#define THREADS_PER_BLOCK_64 64
#define ILP_LEVEL_64 4
#define GRID_DIM_Y_64 1

#define WARPS_PER_BLOCK_128 4
#define THREADS_PER_BLOCK_128 128
#define ILP_LEVEL_128 4
#define GRID_DIM_Y_128 1

#define WARPS_PER_BLOCK_256 4
#define THREADS_PER_BLOCK_256 128
#define ILP_LEVEL_256 4
#define GRID_DIM_Y_256 2

#define WARPS_PER_BLOCK_512 4
#define THREADS_PER_BLOCK_512 128
#define ILP_LEVEL_512 4
#define GRID_DIM_Y_512 4
#elif TN ==2
#define WARPS_PER_BLOCK_16 1
#define THREADS_PER_BLOCK_16 32
#define ILP_LEVEL_16 2
#define GRID_DIM_Y_16 1

#define WARPS_PER_BLOCK_32 2
#define THREADS_PER_BLOCK_32 64
#define ILP_LEVEL_32 2
#define GRID_DIM_Y_32 1

#define WARPS_PER_BLOCK_64 4
#define THREADS_PER_BLOCK_64 128
#define ILP_LEVEL_64 2
#define GRID_DIM_Y_64 1

#define WARPS_PER_BLOCK_128 4
#define THREADS_PER_BLOCK_128 128
#define ILP_LEVEL_128 2
#define GRID_DIM_Y_128 2

#define WARPS_PER_BLOCK_256 4
#define THREADS_PER_BLOCK_256 128
#define ILP_LEVEL_256 2
#define GRID_DIM_Y_256 4

#define WARPS_PER_BLOCK_512 4
#define THREADS_PER_BLOCK_512 128
#define ILP_LEVEL_512 2
#define GRID_DIM_Y_512 8
#else
#define WARPS_PER_BLOCK_16 2
#define THREADS_PER_BLOCK_16 64
#define ILP_LEVEL_16 1
#define GRID_DIM_Y_16 1

#define WARPS_PER_BLOCK_32 4
#define THREADS_PER_BLOCK_32 128
#define ILP_LEVEL_32 1
#define GRID_DIM_Y_32 1

#define WARPS_PER_BLOCK_64 4
#define THREADS_PER_BLOCK_64 128
#define ILP_LEVEL_64 1
#define GRID_DIM_Y_64 2

#define WARPS_PER_BLOCK_128 4
#define THREADS_PER_BLOCK_128 128
#define ILP_LEVEL_128 1
#define GRID_DIM_Y_128 4

#define WARPS_PER_BLOCK_256 4
#define THREADS_PER_BLOCK_256 128
#define ILP_LEVEL_256 1
#define GRID_DIM_Y_256 8

#define WARPS_PER_BLOCK_512 4
#define THREADS_PER_BLOCK_512 128
#define ILP_LEVEL_512 1
#define GRID_DIM_Y_512 16
#endif
using namespace std;
//32 64 128 256 512
class SPMM_N16{
public:
    dim3 gridSize;
    unsigned int n_start;
    unsigned int M;
    unsigned int N;
    unsigned int K;
    void create(unsigned int M_, unsigned int K_, unsigned int N_){
        M = M_;
        K = K_;
        N = N_;
        n_start = 0;
    }
    void run(unsigned int * MatA, float * MatB, float * MatC, uint4 *row_panel_index_array, unsigned int num_row_panels, unsigned int *active_cols_vec, unsigned long long int *sizePtr){
        gridSize = dim3(num_row_panels, GRID_DIM_Y_16);
        SpMM<WARPS_PER_BLOCK_16, ILP_LEVEL_16><<<gridSize, THREADS_PER_BLOCK_16>>>(MatA, MatB, MatC, row_panel_index_array, num_row_panels, N, n_start, active_cols_vec, sizePtr);
    }
};
class SPMM_N32{
public:
    dim3 gridSize;
    unsigned int n_start;
    unsigned int M;
    unsigned int N;
    unsigned int K;
    void create(unsigned int M_, unsigned int K_, unsigned int N_){
        M = M_;
        K = K_;
        N = N_;
        n_start = 0;
    }
    void run(unsigned int * MatA, float * MatB, float * MatC, uint4 *row_panel_index_array, unsigned int num_row_panels, unsigned int *active_cols_vec, unsigned long long int *sizePtr){
        gridSize = dim3(num_row_panels, GRID_DIM_Y_32);
        SpMM<WARPS_PER_BLOCK_32, ILP_LEVEL_32><<<gridSize, THREADS_PER_BLOCK_32>>>(MatA, MatB, MatC, row_panel_index_array, num_row_panels, N, n_start, active_cols_vec, sizePtr);
    }
};
class SPMM_N64{
public:
    dim3 gridSize;
    unsigned int n_start;
    unsigned int M;
    unsigned int N;
    unsigned int K;
    void create(unsigned int M_, unsigned int K_, unsigned int N_){
        M = M_;
        K = K_;
        N = N_;
        n_start = 0;
    }
    void run(unsigned int * MatA, float * MatB, float * MatC, uint4 *row_panel_index_array, unsigned int num_row_panels, unsigned int *active_cols_vec, unsigned long long int *sizePtr){
        gridSize = dim3(num_row_panels, GRID_DIM_Y_64);
        SpMM<WARPS_PER_BLOCK_64, ILP_LEVEL_64><<<gridSize, THREADS_PER_BLOCK_64>>>(MatA, MatB, MatC, row_panel_index_array, num_row_panels, N, n_start, active_cols_vec, sizePtr);
    }
};
class SPMM_N128{
public:
    dim3 gridSize;
    unsigned int n_start;
    unsigned int M;
    unsigned int N;
    unsigned int K;
    void create(unsigned int M_, unsigned int K_, unsigned int N_){
        M = M_;
        K = K_;
        N = N_;
        n_start = 0;
    }
    void run(unsigned int * MatA, float * MatB, float * MatC, uint4 *row_panel_index_array, unsigned int num_row_panels, unsigned int *active_cols_vec, unsigned long long int *sizePtr){
        gridSize = dim3(num_row_panels, GRID_DIM_Y_128);
        SpMM<WARPS_PER_BLOCK_128, ILP_LEVEL_128><<<gridSize, THREADS_PER_BLOCK_128>>>(MatA, MatB, MatC, row_panel_index_array, num_row_panels, N, n_start, active_cols_vec, sizePtr);
    }
};
class SPMM_N256{
public:
    dim3 gridSize;
    unsigned int n_start;
    unsigned int M;
    unsigned int N;
    unsigned int K;
    void create(unsigned int M_, unsigned int K_, unsigned int N_){
        M = M_;
        K = K_;
        N = N_;
        n_start = 0;
    }
    void run(unsigned int * MatA, float * MatB, float * MatC, uint4 *row_panel_index_array, unsigned int num_row_panels, unsigned int *active_cols_vec, unsigned long long int *sizePtr){
        gridSize = dim3(num_row_panels, GRID_DIM_Y_256);
        SpMM<WARPS_PER_BLOCK_256, ILP_LEVEL_256><<<gridSize, THREADS_PER_BLOCK_256>>>(MatA, MatB, MatC, row_panel_index_array, num_row_panels, N, n_start, active_cols_vec, sizePtr);
    }
};
class SPMM_N512{
public:
    dim3 gridSize;
    unsigned int n_start;
    unsigned int M;
    unsigned int N;
    unsigned int K;
    void create(unsigned int M_, unsigned int K_, unsigned int N_){
        M = M_;
        K = K_;
        N = N_;
        n_start = 0;
    }
    void run(unsigned int * MatA, float * MatB, float * MatC, uint4 *row_panel_index_array, unsigned int num_row_panels, unsigned int *active_cols_vec, unsigned long long int *sizePtr){
        gridSize = dim3 (num_row_panels, GRID_DIM_Y_512);
        SpMM<WARPS_PER_BLOCK_512, ILP_LEVEL_512><<<gridSize, THREADS_PER_BLOCK_512>>>(MatA, MatB, MatC, row_panel_index_array, num_row_panels, N, n_start, active_cols_vec, sizePtr);
    }
};
class SPMM_NLARGE{
public:
    dim3 gridSize;
    unsigned int n_start;
    unsigned int M;
    unsigned int N;
    unsigned int K;
    void create(unsigned int M_, unsigned int K_, unsigned int N_){
        M = M_;
        K = K_;
        N = N_;
        n_start = 0;
    }
    void run(unsigned int * MatA, float * MatB, float * MatC, uint4 *row_panel_index_array, unsigned int num_row_panels, unsigned int *active_cols_vec, unsigned long long int *sizePtr){
        gridSize = dim3 (num_row_panels, GRID_DIM_Y_512 * N/512);
        SpMM<WARPS_PER_BLOCK_512, ILP_LEVEL_512><<<gridSize, THREADS_PER_BLOCK_512>>>(MatA, MatB, MatC, row_panel_index_array, num_row_panels, N, n_start, active_cols_vec, sizePtr);
    }
};
struct WarpThreadConfig {
    unsigned int num_warps;
    unsigned int num_thread_blocks;
    unsigned int num_sms;
};
struct Condition{
    unsigned int TILE_M;
    unsigned int ILP_LEVEL;
    unsigned int BLOCK_SIZE;
    unsigned int machine;
};
struct cmpCondition {
    bool operator()(const Condition & a, const Condition & other) const {
        if(a.TILE_M < other.TILE_M){
            return true;
        }
        if(a.TILE_M == other.TILE_M && a.ILP_LEVEL < other.ILP_LEVEL){
            return true;
        }
        if((a.TILE_M == other.TILE_M && a.ILP_LEVEL == other.ILP_LEVEL) && a.BLOCK_SIZE < other.BLOCK_SIZE){
            return true;
        }
        if(((a.TILE_M == other.TILE_M && a.ILP_LEVEL == other.ILP_LEVEL && a.BLOCK_SIZE == other.BLOCK_SIZE) && a.machine < other.machine)){
            return true;
        }
        return false;
    }
};
class SPMM_TC{
public:
    unsigned int M;
    unsigned int N;
    unsigned int K;
    SPMM_N16 spmmN16;
    SPMM_N32 spmmN32;
    SPMM_N64 spmmN64;
    SPMM_N128 spmmN128;
    SPMM_N256 spmmN256;
    SPMM_N512 spmmN512;
    SPMM_NLARGE spmmNlarge;
    uint4 * row_panel_index_array;
    uint4 * row_panel_index_array_device;
    unsigned int numRowPanelsSplit;
    float waves_before_partition;
    float waves_after_partition;
    std::map<Condition, WarpThreadConfig, cmpCondition> conditions;
    SPMM_TC(unsigned int M_, unsigned int K_, unsigned int N_, unsigned int *blockedRowPtr){
        M = M_;
        K = K_;
        N = N_;
        spmmN16.create(M, K, 16);
        spmmN32.create(M, K, 32);
        spmmN64.create(M, K, 64);
        spmmN128.create(M, K, 128);
        spmmN256.create(M, K, 256);
        spmmN512.create(M, K, 512);
        spmmNlarge.create(M, K, N);
        initialize_conditions();
        partition(blockedRowPtr);
    }
    void initialize_conditions(){
        Condition condition;
        condition.TILE_M = 16;
        condition.ILP_LEVEL = 1;
        condition.BLOCK_SIZE = 2;
        condition.machine = 0;
        WarpThreadConfig warpThreadConfig;
        warpThreadConfig.num_warps = 48;
        warpThreadConfig.num_thread_blocks = 24;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 1;
        condition.BLOCK_SIZE = 4;
        condition.machine = 0;
        warpThreadConfig.num_warps = 48;
        warpThreadConfig.num_thread_blocks = 12;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 1;
        condition.BLOCK_SIZE = 4;
        condition.machine = 0;
        warpThreadConfig.num_warps = 40;
        warpThreadConfig.num_thread_blocks = 10;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 1;
        condition.BLOCK_SIZE = 2;
        condition.machine = 0;
        warpThreadConfig.num_warps = 38;
        warpThreadConfig.num_thread_blocks = 19;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 1;
        condition.machine = 0;
        warpThreadConfig.num_warps = 25;
        warpThreadConfig.num_thread_blocks = 25;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 2;
        condition.machine = 0;
        warpThreadConfig.num_warps = 36;
        warpThreadConfig.num_thread_blocks = 18;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 4;
        condition.machine = 0;
        warpThreadConfig.num_warps = 36;
        warpThreadConfig.num_thread_blocks = 9;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 1;
        condition.machine = 0;
        warpThreadConfig.num_warps = 19;
        warpThreadConfig.num_thread_blocks = 19;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 2;
        condition.machine = 0;
        warpThreadConfig.num_warps = 30;
        warpThreadConfig.num_thread_blocks = 15;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 4;
        condition.machine = 0;
        warpThreadConfig.num_warps = 36;
        warpThreadConfig.num_thread_blocks = 9;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 1;
        condition.machine = 0;
        warpThreadConfig.num_warps = 19;
        warpThreadConfig.num_thread_blocks = 19;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 2;
        condition.machine = 0;
        warpThreadConfig.num_warps = 26;
        warpThreadConfig.num_thread_blocks = 13;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 4;
        condition.machine = 0;
        warpThreadConfig.num_warps = 32;
        warpThreadConfig.num_thread_blocks = 8;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 1;
        condition.machine = 0;
        warpThreadConfig.num_warps = 15;
        warpThreadConfig.num_thread_blocks = 15;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 2;
        condition.machine = 0;
        warpThreadConfig.num_warps = 22;
        warpThreadConfig.num_thread_blocks = 11;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 4;
        condition.machine = 0;
        warpThreadConfig.num_warps = 28;
        warpThreadConfig.num_thread_blocks = 7;
        warpThreadConfig.num_sms = 108;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 1;
        condition.BLOCK_SIZE = 2;
        condition.machine = 1;
        warpThreadConfig.num_warps = 30;
        warpThreadConfig.num_thread_blocks = 15;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 1;
        condition.BLOCK_SIZE = 4;
        condition.machine = 1;
        warpThreadConfig.num_warps = 44;
        warpThreadConfig.num_thread_blocks = 11;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 1;
        condition.BLOCK_SIZE = 4;
        condition.machine = 1;
        warpThreadConfig.num_warps = 36;
        warpThreadConfig.num_thread_blocks = 9;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 1;
        condition.BLOCK_SIZE = 2;
        condition.machine = 1;
        warpThreadConfig.num_warps = 22;
        warpThreadConfig.num_thread_blocks = 11;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 1;
        condition.machine = 1;
        warpThreadConfig.num_warps = 15;
        warpThreadConfig.num_thread_blocks = 15;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 2;
        condition.machine = 1;
        warpThreadConfig.num_warps = 22;
        warpThreadConfig.num_thread_blocks = 11;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 4;
        condition.machine = 1;
        warpThreadConfig.num_warps = 32;
        warpThreadConfig.num_thread_blocks = 8;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;
        condition.TILE_M = 32;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 1;
        condition.machine = 1;
        warpThreadConfig.num_warps = 11;
        warpThreadConfig.num_thread_blocks = 11;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 2;
        condition.machine = 1;
        warpThreadConfig.num_warps = 18;
        warpThreadConfig.num_thread_blocks = 9;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 4;
        condition.machine = 1;
        warpThreadConfig.num_warps = 24;
        warpThreadConfig.num_thread_blocks = 6;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 1;
        condition.machine = 1;
        warpThreadConfig.num_warps = 11;
        warpThreadConfig.num_thread_blocks = 11;
        warpThreadConfig.num_sms = 128;

        conditions[condition] = warpThreadConfig;
        condition.TILE_M = 16;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 2;
        condition.machine = 1;
        warpThreadConfig.num_warps = 16;
        warpThreadConfig.num_thread_blocks = 8;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 4;
        condition.machine = 1;
        warpThreadConfig.num_warps = 16;
        warpThreadConfig.num_thread_blocks = 4;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 1;
        condition.machine = 1;
        warpThreadConfig.num_warps = 9;
        warpThreadConfig.num_thread_blocks = 9;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 2;
        condition.machine = 1;
        warpThreadConfig.num_warps = 12;
        warpThreadConfig.num_thread_blocks = 6;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 4;
        condition.machine = 1;
        warpThreadConfig.num_warps = 16;
        warpThreadConfig.num_thread_blocks = 4;
        warpThreadConfig.num_sms = 128;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 1;
        condition.BLOCK_SIZE = 2;
        condition.machine = 2;
        warpThreadConfig.num_warps = 30;
        warpThreadConfig.num_thread_blocks = 15;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 1;
        condition.BLOCK_SIZE = 4;
        condition.machine = 2;
        warpThreadConfig.num_warps = 44;
        warpThreadConfig.num_thread_blocks = 11;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;


        condition.TILE_M = 16;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 1;
        condition.machine = 2;
        warpThreadConfig.num_warps = 15;
        warpThreadConfig.num_thread_blocks = 15;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 2;
        condition.machine = 2;
        warpThreadConfig.num_warps = 22;
        warpThreadConfig.num_thread_blocks = 11;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 4;
        condition.machine = 2;
        warpThreadConfig.num_warps = 32;
        warpThreadConfig.num_thread_blocks = 8;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 2;
        condition.BLOCK_SIZE = 1;
        condition.machine = 2;
        warpThreadConfig.num_warps = 26;
        warpThreadConfig.num_thread_blocks = 26;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;


        condition.TILE_M = 16;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 1;
        condition.machine = 2;
        warpThreadConfig.num_warps = 11;
        warpThreadConfig.num_thread_blocks = 11;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;


        condition.TILE_M = 16;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 2;
        condition.machine = 2;
        warpThreadConfig.num_warps = 16;
        warpThreadConfig.num_thread_blocks = 8;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 16;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 4;
        condition.machine = 2;
        warpThreadConfig.num_warps = 16;
        warpThreadConfig.num_thread_blocks = 4;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 1;
        condition.machine = 2;
        warpThreadConfig.num_warps = 21;
        warpThreadConfig.num_thread_blocks = 21;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 2;
        condition.machine = 2;
        warpThreadConfig.num_warps = 30;
        warpThreadConfig.num_thread_blocks = 15;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;

        condition.TILE_M = 32;
        condition.ILP_LEVEL = 4;
        condition.BLOCK_SIZE = 4;
        condition.machine = 2;
        warpThreadConfig.num_warps = 32;
        warpThreadConfig.num_thread_blocks = 8;
        warpThreadConfig.num_sms = 144;
        conditions[condition] = warpThreadConfig;
    }
    void partition(unsigned int *blockedRowPtr){
        unsigned int num_sms;
        unsigned int num_thread_blocks;
        WarpThreadConfig config;
        unsigned int num_blocks_dimension_n;
        unsigned int ilp_level;
        if(N == 16){
            Condition condition;
            condition.TILE_M = TM;
            condition.ILP_LEVEL = ILP_LEVEL_16;
            condition.BLOCK_SIZE = WARPS_PER_BLOCK_16;
            condition.machine = Machine;
            config = conditions[condition];
            num_sms = config.num_sms;
            num_thread_blocks = config.num_thread_blocks;
            num_blocks_dimension_n = 1;
        }else if(N == 32){
            Condition condition;
            condition.TILE_M = TM;
            condition.ILP_LEVEL = ILP_LEVEL_32;
            condition.BLOCK_SIZE = WARPS_PER_BLOCK_32;
            condition.machine = Machine;
            config = conditions[condition];
            num_sms = config.num_sms;
            num_thread_blocks = config.num_thread_blocks;
            num_blocks_dimension_n = 1;
        }else if(N == 64){
            Condition condition;
            condition.TILE_M = TM;
            condition.ILP_LEVEL = ILP_LEVEL_64;
            condition.BLOCK_SIZE = WARPS_PER_BLOCK_64;
            condition.machine = Machine;
            config = conditions[condition];
            num_sms = config.num_sms;
            num_thread_blocks = config.num_thread_blocks;
            if(condition.ILP_LEVEL == 1){
                num_blocks_dimension_n = 2;
            }else{
                num_blocks_dimension_n = 1;
            }
        }else if(N == 128){
            Condition condition;
            condition.TILE_M = TM;
            condition.ILP_LEVEL = ILP_LEVEL_128;
            condition.BLOCK_SIZE = WARPS_PER_BLOCK_128;
            condition.machine = Machine;
            config = conditions[condition];
            num_sms = config.num_sms;
            num_thread_blocks = config.num_thread_blocks;
            ilp_level = condition.ILP_LEVEL;
            num_blocks_dimension_n = 128/(4 * ilp_level * 8);
        }else if(N == 256){
            Condition condition;
            condition.TILE_M = TM;
            condition.ILP_LEVEL = ILP_LEVEL_256;
            condition.BLOCK_SIZE = WARPS_PER_BLOCK_256;
            condition.machine = Machine;
            config = conditions[condition];
            num_sms = config.num_sms;
            num_thread_blocks = config.num_thread_blocks;
            ilp_level = condition.ILP_LEVEL;
            num_blocks_dimension_n = 256/(4 * ilp_level * 8);
        }else if(N == 512){
            Condition condition;
            condition.TILE_M = TM;
            condition.ILP_LEVEL = ILP_LEVEL_512;
            condition.BLOCK_SIZE = WARPS_PER_BLOCK_512;
            condition.machine = Machine;
            config = conditions[condition];
            num_sms = config.num_sms;
            num_thread_blocks = config.num_thread_blocks;
            ilp_level = condition.ILP_LEVEL;
            num_blocks_dimension_n = 512/(4 * ilp_level * 8);
        }else{
            Condition condition;
            condition.TILE_M = TM;
            condition.ILP_LEVEL = 2;
            condition.BLOCK_SIZE = 4;
            condition.machine = Machine;
            config = conditions[condition];
            num_sms = config.num_sms;
            num_thread_blocks = config.num_thread_blocks;
            ilp_level = 2;
            num_blocks_dimension_n = ((N - 1)/(4 * ilp_level * 8) + 1);
        }
        vector<uint4> row_panel_index_vec;
        unsigned int num_row_panels = (M - 1)/TM + 1;
        waves_before_partition = (num_row_panels * num_blocks_dimension_n * 1.0f)/(num_sms * num_thread_blocks);
        unsigned int num_waves = ((num_row_panels * num_blocks_dimension_n - 1)/(num_sms * num_thread_blocks) + 1);
        unsigned int average_blocks_each_row_panel = max(blockedRowPtr[num_row_panels]/num_row_panels, 1);
        for(unsigned int i=0; i<num_row_panels; ++i){
            unsigned int start = blockedRowPtr[i];
            unsigned int end = blockedRowPtr[i + 1];
            if(start == end){
                continue;
            }
            unsigned int num_blocks = end - start;
            unsigned int num_loads = (num_blocks - 1)/average_blocks_each_row_panel + 1;
            unsigned int ratio = (num_loads - 1)/num_waves + 1;
            unsigned int chunk_size = (num_blocks - 1)/ratio + 1;
            if(ratio > 1){
                unsigned int k_iters = ratio;
                for(unsigned int k=0; k<k_iters; ++k){
                    uint4 index;
                    unsigned int start_sub = start + k * chunk_size;
                    unsigned int end_sub = start + (k + 1) * chunk_size;
                    end_sub = min(end_sub, end);
                    index.x = start_sub;
                    index.y = end_sub;
                    index.z = i;
                    index.w = 1;
                    row_panel_index_vec.push_back(index);
                }
            }else{
                uint4 index;
                index.x = start;
                index.y = end;
                index.z = i;
                index.w = 0;
                row_panel_index_vec.push_back(index);
            }
        }
        numRowPanelsSplit = row_panel_index_vec.size();
        waves_after_partition = (numRowPanelsSplit * num_blocks_dimension_n * 1.0f)/(num_sms * num_thread_blocks);
        uint4 * row_panel_index_array = new uint4 [row_panel_index_vec.size()];
        for(unsigned int i=0; i<row_panel_index_vec.size(); ++i){
            row_panel_index_array[i] = row_panel_index_vec[i];
        }
        chkerr(cudaMalloc(&row_panel_index_array_device, row_panel_index_vec.size() * sizeof(uint4)), __LINE__);
        chkerr(cudaMemcpy(row_panel_index_array_device, row_panel_index_array, row_panel_index_vec.size() * sizeof(uint4),
                          cudaMemcpyHostToDevice), __LINE__);
    }
    void run(unsigned int * MatA, float * MatB, float * MatC, unsigned int * active_cols_vec, unsigned long long int *sizePtr){
        if(N == 16){
            spmmN16.run(MatA, MatB, MatC, row_panel_index_array_device, numRowPanelsSplit, active_cols_vec, sizePtr);
        }else if(N == 32){
            spmmN32.run(MatA, MatB, MatC, row_panel_index_array_device, numRowPanelsSplit, active_cols_vec, sizePtr);
        }else if(N == 64){
            spmmN64.run(MatA, MatB, MatC, row_panel_index_array_device, numRowPanelsSplit, active_cols_vec, sizePtr);
        }else if(N == 128){
            spmmN128.run(MatA, MatB, MatC, row_panel_index_array_device, numRowPanelsSplit, active_cols_vec, sizePtr);
        }else if(N == 256){
            spmmN256.run(MatA, MatB, MatC, row_panel_index_array_device, numRowPanelsSplit, active_cols_vec, sizePtr);
        }else if(N == 512){
            spmmN512.run(MatA, MatB, MatC, row_panel_index_array_device, numRowPanelsSplit, active_cols_vec, sizePtr);
        }else{
            spmmNlarge.run(MatA, MatB, MatC, row_panel_index_array_device, numRowPanelsSplit, active_cols_vec, sizePtr);
        }
    }
};
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
float GPU_CSR_cuSPARSE_NEW(unsigned int * hA_csrOffsets, unsigned int *  hA_columns, COMPUTETYPE * hA_values, unsigned int A_nnz,
                           COMPUTETYPE *hB, float *hC, uint32_t m, uint32_t k , uint32_t n, int iters, float alpha=1.0, float beta=0.0)
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
    unsigned int  *dA_csrOffsets, *dA_columns;
    COMPUTETYPE* *dA_values, *dB;
    float  *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,(A_num_rows + 1) * sizeof(unsigned int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(unsigned int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(COMPUTETYPE))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(COMPUTETYPE)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,(A_num_rows + 1) * sizeof(unsigned int),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(unsigned int),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(COMPUTETYPE),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(COMPUTETYPE),cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),cudaMemcpyHostToDevice) )
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
                                      CUSPARSE_INDEX_BASE_ZERO,  CUDA_R_32F))

    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, k, n, n, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW) )

    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, m, n, n, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_CSR_ALG2, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
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
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    // CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // cpy results to host
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),cudaMemcpyDeviceToHost) )
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

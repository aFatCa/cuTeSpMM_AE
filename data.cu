#include "IO.cu"
#include <cassert>
#include <unordered_map>
#include <array>
#include <set>
#include <omp.h>
#include <math.h>
#include <bitset>
#include <type_traits>
#include <typeinfo>
#define Divide 4
#define NUM_THREADS 32
#define MAX_THREADS 64
using namespace std;
unsigned int round(unsigned int count, unsigned int round){
    return ((count - 1)/round + 1) * round;
}
unsigned long long int pattern_encode(float *vals){
    unsigned long long int pattern = 0;
    for(unsigned int i=0; i<64; ++i){
        unsigned long long int index = 1;
        if(vals[i] != 0.0f){
            pattern = pattern | (index << i);
        }
    }
    return pattern;
}
void set1(unsigned long long int &pattern, unsigned int index){
    unsigned long long int pos = 1;
    pos = (pos << index);
    pattern = pattern | pos;
}

void set_pattern(unsigned long long int *patterns, unsigned int pattern_id, unsigned int brick_m,
                 unsigned int brick_k, unsigned &nnz1, unsigned &nnz2){
    unsigned int pos = brick_m * inst_k + brick_k;
    unsigned long long int index = 1;
    patterns[pattern_id] = (patterns[pattern_id] | (index << pos));
    (brick_m < 8) ? nnz1++ : nnz2++;
}
void print_pattern(unsigned long long int pattern){
    for(int i=0;i<inst_m;++i){
        for(int j=0;j<inst_k;++j){
            unsigned int pos = i * inst_k + j;
            if((pattern >> pos) &1){
                std::cout<<"1"<<",";
            }else{
                std::cout<<"0"<<",";
            }
        }
        std::cout<<std::endl;
    }
}
class HRPB{
public:
    unsigned int *cols;
    unsigned int *rowPtr;
    float *vals;
    unsigned int M;
    unsigned int K;
    unsigned int numRowPanels;
    unsigned int *blocksData;
    unsigned long long int dataStructSize;
    unsigned int num_bricks;
    unsigned int *blocksDevice;
    unsigned int *blockedRowPtr;
    unsigned int *blockedRowPtrDevice;
    unsigned long long int *sizePtr;
    unsigned long long int *sizePtrDevice;
    unsigned int numActiveCols;
    unsigned int *activeCols;
    unsigned int *activeColsDevice;
    float averageNNZcol;
    unsigned int numNNZ;
    float block_density;
    unsigned int atomic_blocks;
    float atomic_ratio;
    float waves;
    std::vector< vector<unsigned int> > blocks_nnz_counter_vec;
    HRPB(unsigned int M_, unsigned int K_, unsigned int *rowPtr_, unsigned int *cols_, float *vals_){
        num_bricks = 0;
        rowPtr = rowPtr_;
        cols = cols_;
        vals = vals_;
        M = M_;
        K = K_;
        numRowPanels = M / TM;
        numNNZ = 0;
        num_bricks = 0;
        numActiveCols = 0;
        sizePtr = new unsigned long long int[(M/TM) * (K/TK) + 1];
        DATASTRUCT();
    }
    void DATASTRUCT() {
        unsigned int *helper = new unsigned int[K * MAX_THREADS];
        unsigned int *active_col_vec = new unsigned int [K * MAX_THREADS];
        unsigned int *cp_buffer = new unsigned int [TM * MAX_THREADS];
        unsigned int *col_map_vec = new unsigned int [K * MAX_THREADS];
        blockedRowPtr = new unsigned int[numRowPanels + 1];
        memset(blockedRowPtr, 0, (numRowPanels + 1) * sizeof(unsigned int));
        unsigned int *num_blocks_trunk_list = new unsigned int [MAX_THREADS + 1];
        unsigned long long int *size_trunk_list = new unsigned long long int[MAX_THREADS + 1];
        memset(num_blocks_trunk_list, 0, (MAX_THREADS + 1) * sizeof(unsigned int));
        memset(size_trunk_list, 0, (MAX_THREADS + 1) * sizeof(unsigned long long int));
        unsigned int trunk_size = (numRowPanels - 1) / NUM_THREADS + 1;
        unsigned int num_threads_required = (numRowPanels - 1)/trunk_size + 1;
#pragma omp parallel num_threads(num_threads_required)
        {
            int thread_id = omp_get_thread_num();
            unsigned int *helper_local = &helper[thread_id * K];
            unsigned int *active_col_vec_local = &active_col_vec[thread_id * K];
            unsigned int *cp_buffer_local = &cp_buffer[thread_id * TM];
            unsigned int *col_map_vec_local = &col_map_vec[thread_id * K];
            memset(helper_local, 0, K * sizeof(unsigned int));
            unsigned long long int *patterns = new unsigned long long [TM/inst_m];
            for(unsigned int i = 0; i < trunk_size; ++i) {
                if(thread_id * trunk_size + i >= numRowPanels) {
                    break;
                }
                unsigned int row_panel_index = thread_id * trunk_size + i;
                unsigned int row_start = row_panel_index * TM;
                unsigned int row_end = (row_panel_index + 1) * TM;
                unsigned int len = 0;
                for(unsigned int ii = row_start; ii < row_end; ++ii) {
                    unsigned int col_start = rowPtr[ii];
                    unsigned int col_end = rowPtr[ii + 1];
                    for (unsigned int j = col_start; j < col_end; ++j) {
                        unsigned int col = cols[j];
                        if (helper_local[col] == 0) {
                            active_col_vec_local[len] = col;
                            len++;
                            helper_local[col] = 1;
                        }
                    }
                }
                unsigned int num_blocks_row_panel;
                if (len == 0) {
                    num_blocks_row_panel = 0;
                } else {
                    num_blocks_row_panel = (len - 1) / TK + 1;
                }
                //__sync_fetch_and_add(&numActiveCols, len);
                blockedRowPtr[row_panel_index + 1] = num_blocks_row_panel;
                num_blocks_trunk_list[thread_id + 1] += num_blocks_row_panel;
                for (unsigned int ii = 0; ii < len; ++ii) {
                    unsigned int col = active_col_vec_local[ii];
                    helper_local[col] = 0;
                }
            }
            unsigned int num_blocks_my_trunk = num_blocks_trunk_list[thread_id + 1];
            unsigned int *work_space = new unsigned int[num_blocks_my_trunk * (BLK_MEM_SIZE)];
            memset(work_space, 0, num_blocks_my_trunk * (BLK_MEM_SIZE) * sizeof(unsigned int));
#pragma omp barrier
#pragma omp master
            {
                for(unsigned int i=0; i<num_threads_required; ++i){
                    num_blocks_trunk_list[i + 1] += num_blocks_trunk_list[i];
                }
                activeCols = new unsigned int [num_blocks_trunk_list[num_threads_required] * TK];
                memset(activeCols, 0, num_blocks_trunk_list[num_threads_required] * TK * sizeof(unsigned int));
            }
#pragma omp barrier
            unsigned long long int *size_ptr_local = new unsigned long long int [num_blocks_my_trunk + 1];
            memset(size_ptr_local, 0, (num_blocks_my_trunk + 1) * sizeof(unsigned long long int));
            unsigned int block_index = 0;
            unsigned long long int block_addr = 0;
            unsigned long long int pre_block_addr = 0;
            unsigned int col_addr = num_blocks_trunk_list[thread_id] * TK;
            if(thread_id * trunk_size < numRowPanels){
                blockedRowPtr[thread_id * trunk_size] = num_blocks_trunk_list[thread_id];
            }
#pragma unroll
            for(unsigned int i = 0; i < trunk_size; ++i) {
                if(thread_id * trunk_size + i >= numRowPanels){
                    break;
                }
                if(thread_id != num_threads_required - 1){
                    if(i < trunk_size - 1){
                        blockedRowPtr[thread_id * trunk_size + i + 1] += blockedRowPtr[thread_id * trunk_size + i];
                    }
                }else{
                    blockedRowPtr[thread_id * trunk_size + i + 1] += blockedRowPtr[thread_id * trunk_size + i];
                }
                unsigned int row_panel_index = thread_id * trunk_size + i;
                unsigned int row_start = row_panel_index * TM;
                unsigned int row_end = (row_panel_index + 1) * TM;
                unsigned int len = 0;
                for(unsigned int ii = row_start; ii < row_end; ++ii) {
                    unsigned int col_start = rowPtr[ii];
                    unsigned int col_end = rowPtr[ii + 1];
                    cp_buffer_local[ii - row_start] = rowPtr[ii];
                    for(unsigned int j = col_start; j < col_end; ++j) {
                        unsigned int col = cols[j];
                        if(helper_local[col] == 0) {
                            active_col_vec_local[len] = col;
                            len++;
                        }
                        helper_local[col]++;
                    }
                }
                if(len == 0){
                    continue;
                }
                std::sort(active_col_vec_local, active_col_vec_local + len);
                for(unsigned int ii=0; ii<len; ++ii) {
                    unsigned int col = active_col_vec_local[ii];
                    activeCols[col_addr + ii] = col;
                    col_map_vec_local[col] = ii;
                }
                col_addr += ((len - 1)/TK + 1)*TK;
                unsigned int num_blocks_row_panel = (len - 1) / TK + 1;
                for(unsigned int ii = 0; ii < num_blocks_row_panel; ii++){
                    unsigned int *block_data = &work_space[block_addr];
                    unsigned int nnz_counters = 0;
#pragma unroll
                    for(unsigned int k=0; k<TK; ++k){
                        if(ii * TK + k < len){
                            unsigned int col = active_col_vec_local[ii * TK + k];
                            nnz_counters += helper_local[col];
                            if(ii != 0){
                                work_space[pre_block_addr+k+NEXT_COLS_OFFSET] = col;
                            }
                        }
                    }
                    unsigned int block_data_size = round(nnz_counters, 2) + MetaDataSize;
                    block_data[FETCH_ITERS_OFFSET] = (block_data_size - 1)/128 + 1;
                    block_data_size = block_data[FETCH_ITERS_OFFSET] * 128;
                    if(ii != 0){
                        work_space[pre_block_addr + NEXT_FETCH_ITERS_OFFSET] = (block_data_size - 1)/128 + 1;
                    }
                    unsigned int counter = 0;
                    unsigned long long int col_ptr_pattern = 0;
                    float *block_nnz_list = reinterpret_cast<float *>(&block_data[MetaDataSize]);
                    ushort temp_nnz_counters[NUM_TK_BRICKS + 1] = {0};
#pragma unroll
                    for(unsigned int brick_col_index = 0; brick_col_index < NUM_TK_BRICKS; ++brick_col_index){
                        memset(patterns, 0, (TM/inst_m) * sizeof(unsigned long long int));
                        unsigned int brick_pattern_index = 0;
                        unsigned int nnz_pattern = 0;
#pragma unroll
                        for(unsigned int j=0; j<TM/inst_m; ++j){
                            unsigned int brick_row = j;
                            bool set_flag = false;
                            unsigned int nnz1 = 0;
                            unsigned int nnz2 = 0;
                            for(unsigned int jj=0; jj<inst_m; ++jj){
                                unsigned int brick_m = jj;
                                unsigned int col_start = cp_buffer_local[j * inst_m + jj];
                                unsigned int col_end = rowPtr[j*inst_m + jj + row_start + 1];
                                for(unsigned int col_index = col_start; col_index<col_end; ++col_index){
                                    unsigned int col = cols[col_index];
                                    col = col_map_vec_local[col];
                                    unsigned int brick_k = col%inst_k;
                                    unsigned int block_id = col/TK;
                                    unsigned int brick_id = (col%TK)/inst_k;
                                    if(block_id > ii || brick_id > brick_col_index){
                                        break;
                                    }
                                    unsigned int set_index = brick_id * 4 + brick_row;
                                    set1(col_ptr_pattern, set_index);
                                    float value = vals[col_index];
                                    block_nnz_list[counter] = value;
                                    counter++;
                                    set_pattern(patterns, brick_pattern_index, brick_m, brick_k, nnz1, nnz2);
                                    temp_nnz_counters[brick_col_index+1]++;
                                    cp_buffer_local[j * inst_m + jj]++;
                                    set_flag = true;
                                }
                            }
                            if(set_flag){
                                //__sync_fetch_and_add(&num_bricks, 1);
                                nnz2 = (nnz2 << 8);
                                nnz1 = (nnz1 << (brick_pattern_index * 16));
                                nnz2 = (nnz2 << (brick_pattern_index * 16));
                                nnz_pattern = nnz_pattern | nnz1;
                                nnz_pattern = nnz_pattern | nnz2;
                                brick_pattern_index++;
                            }
                        }
                        block_data[PATTERN_NNZ_COUNTER_OFFSET + brick_col_index] = nnz_pattern;
                        unsigned long long int *brick_patterns =
                                reinterpret_cast<unsigned long long int*>(&block_data[BRICK_PATTERN_OFFSET + brick_col_index * r_tiles * 2]);
                        for(int j=0; j<TM/inst_m; ++j){
                            brick_patterns[j] = patterns[j];
                        }
                    }
                    for(unsigned int jj=0;jj<NUM_TK_BRICKS;++jj){
                        temp_nnz_counters[jj + 1] += temp_nnz_counters[jj];
                    }
                    memcpy(&block_data[NNZ_COUNTER_OFFSET], temp_nnz_counters, sizeof(ushort) * NUM_TK_BRICKS);
                    memcpy(&block_data[COL_PTR_OFFSET], &col_ptr_pattern, sizeof(unsigned long long int));
                    pre_block_addr = block_addr;
                    block_addr += block_data_size;
                    size_ptr_local[block_index + 1] = block_addr;
                    block_index ++;
                }
                for(unsigned int ii = 0; ii < len; ++ii) {
                    unsigned int col = active_col_vec_local[ii];
                    helper_local[col] = 0;
                }
            }
            size_trunk_list[thread_id + 1] = block_addr;
#pragma omp barrier
#pragma omp master
            {
                for(unsigned int i=0; i<num_threads_required; ++i){
                    size_trunk_list[i + 1] += size_trunk_list[i];
                }
                blocksData = new unsigned int [size_trunk_list[num_threads_required]];
            }
#pragma omp barrier
            for(unsigned int i=0; i<num_blocks_my_trunk + 1; ++i){
                unsigned long long int addr = size_ptr_local[i];
                size_ptr_local[i] += size_trunk_list[thread_id];
                (reinterpret_cast<unsigned long long int*>(&work_space[pre_block_addr + NEXT_BLK_MEM_ADDR_OFFSET]))[0] = size_ptr_local[i];
                pre_block_addr = addr;
            }
            if(thread_id != (num_threads_required - 1)){
                memcpy(&sizePtr[num_blocks_trunk_list[thread_id]], size_ptr_local, num_blocks_my_trunk*sizeof(unsigned long long int));
            }else{
                memcpy(&sizePtr[num_blocks_trunk_list[thread_id]], size_ptr_local, (num_blocks_my_trunk + 1)*sizeof(unsigned long long int));
            }
            memcpy(&blocksData[size_trunk_list[thread_id]], work_space, block_addr*sizeof(unsigned int));
            delete[] work_space;
            delete[] size_ptr_local;
        }
        chkerr(cudaMalloc(&blockedRowPtrDevice, (numRowPanels + 1)*sizeof(unsigned int)), __LINE__);
        chkerr(cudaMemcpy(blockedRowPtrDevice, blockedRowPtr, (numRowPanels + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice), __LINE__);
        chkerr(cudaMalloc(&blocksDevice, size_trunk_list[num_threads_required] * sizeof(unsigned int)), __LINE__);
        chkerr(cudaMalloc(&sizePtrDevice, blockedRowPtr[numRowPanels] * sizeof(unsigned long long int)), __LINE__);
        chkerr(cudaMemcpy(blocksDevice, blocksData, size_trunk_list[num_threads_required]*sizeof(unsigned int), cudaMemcpyHostToDevice), __LINE__);
        chkerr(cudaMemcpy(sizePtrDevice, sizePtr, blockedRowPtr[numRowPanels] * sizeof(unsigned long long int), cudaMemcpyHostToDevice), __LINE__);
        chkerr(cudaMalloc(&activeColsDevice, blockedRowPtr[numRowPanels]*TK*sizeof(unsigned int)), __LINE__);
        chkerr(cudaMemcpy(activeColsDevice, activeCols, blockedRowPtr[numRowPanels]*TK*sizeof(unsigned int), cudaMemcpyHostToDevice), __LINE__);
    }
    void freeDevice(){
        cudaFree(blockedRowPtrDevice);
        cudaFree(blocksDevice);
    }
};

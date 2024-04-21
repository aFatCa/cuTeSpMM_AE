#include "data.cu"
#define PADDING 4
using namespace nvcuda;
__forceinline__ __device__ unsigned lane_id_(){
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
__forceinline__ __device__ unsigned int BitCount(unsigned int i){
    i = i - ((i >> 1) & 0x55555555);        // add pairs of bits
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);  // quads
    i = (i + (i >> 4)) & 0x0F0F0F0F;        // groups of 8
    i *= 0x01010101;                        // horizontal sum of bytes
    return  i >> 24;
}
template <unsigned int ILP_LEVEL>
__forceinline__ __device__ void COMPUTE(unsigned int col_pattern, float * nnzs, unsigned int pre_bits_mask,
                                            unsigned int count, unsigned long long int *brick_patterns, unsigned int lane_id,
                                            uint32_t * frag_A, uint32_t* frag_B, float *frag_D , float *MatB_SEM, unsigned int counter_pattern){
    unsigned long long int pattern = brick_patterns[0];
    unsigned int pattern_first = pattern & 0xffffffff;
    unsigned int pattern_second = (pattern >> 32) & 0xffffffff;
    unsigned int index1 = BitCount(pattern_first & pre_bits_mask);
    unsigned int num_nnz_first = (counter_pattern & 0x000000ff);
    unsigned int index2 = BitCount(pattern_second & pre_bits_mask) + num_nnz_first;
    float val1 = ((pattern_first >> lane_id) & 0b1) ? nnzs[count + index1] : 0.0f;
    float val2 = ((pattern_second >> lane_id) & 0b1) ? nnzs[count + index2] : 0.0f;
    asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(val1));
    asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(val2));
    uint32_t const * A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
    uint32_t const * B = reinterpret_cast<uint32_t const *>(&frag_B[0]);
    #if TM == 16
    #pragma unroll
        for(unsigned int ilp = 0; ilp<ILP_LEVEL; ++ilp){
            asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[ilp]) :
                    "f"(MatB_SEM[ilp * inst_n])
                    );
            asm volatile(
                    "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                    : "=f"(frag_D[0 + ilp * 4]), "=f"(frag_D[1 + ilp * 4]), "=f"(frag_D[2 + ilp * 4]), "=f"(frag_D[3 + ilp * 4])
                    : "r"(A[0]), "r"(A[1]),
            "r"(B[ilp]),
            "f"(frag_D[0 + ilp * 4]), "f"(frag_D[1 + ilp * 4]), "f"(frag_D[2 + ilp * 4]), "f"(frag_D[3 + ilp * 4])
                    );
        }
    #else
        unsigned int num_nnz_second = ((counter_pattern >> 8) & 0x000000ff);
        count = count + num_nnz_first + num_nnz_second;
        switch (col_pattern) {
            case 0b0001:
                #pragma unroll
                for(unsigned int ilp = 0; ilp<ILP_LEVEL; ++ilp){
                    asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[ilp]) :
                            "f"(MatB_SEM[ilp * inst_n])
                            );
                    asm volatile(
                            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                            : "=f"(frag_D[0 + ilp * 4]), "=f"(frag_D[1 + ilp * 4]), "=f"(frag_D[2 + ilp * 4]), "=f"(frag_D[3 + ilp * 4])
                            : "r"(A[0]), "r"(A[1]),
                    "r"(B[ilp]),
                    "f"(frag_D[0 + ilp * 4]), "f"(frag_D[1 + ilp * 4]), "f"(frag_D[2 + ilp * 4]), "f"(frag_D[3 + ilp * 4])
                            );
                }
                break;
            case 0b0010:
                #pragma unroll
                for(unsigned int ilp = 0; ilp<ILP_LEVEL; ++ilp){
                    asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[ilp]) :
                            "f"(MatB_SEM[ilp * inst_n])
                            );
                    asm volatile(
                            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                            : "=f"(frag_D[ILP_LEVEL * 4 + 0 + ilp * 4]), "=f"(frag_D[ILP_LEVEL * 4 + 1 + ilp * 4]), "=f"(frag_D[ILP_LEVEL * 4 + 2 + ilp * 4]), "=f"(frag_D[ILP_LEVEL * 4 + 3 + ilp * 4])
                            : "r"(A[0]), "r"(A[1]),
                    "r"(B[ilp]),
                    "f"(frag_D[ILP_LEVEL * 4 + 0 + ilp * 4]), "f"(frag_D[ILP_LEVEL * 4 + 1 + ilp * 4]), "f"(frag_D[ILP_LEVEL * 4 + 2 + ilp * 4]), "f"(frag_D[ILP_LEVEL * 4 + 3 + ilp * 4])
                            );
                }
                break;
            case 0b0011:
                #pragma unroll
                for(unsigned int ilp = 0; ilp<ILP_LEVEL; ++ilp){
                    asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[ilp]) :
                            "f"(MatB_SEM[ilp * inst_n])
                            );
                    asm volatile(
                            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                            : "=f"(frag_D[0 + ilp * 4]), "=f"(frag_D[1 + ilp * 4]), "=f"(frag_D[2 + ilp * 4]), "=f"(frag_D[3 + ilp * 4])
                            : "r"(A[0]), "r"(A[1]),
                    "r"(B[ilp]),
                    "f"(frag_D[0 + ilp * 4]), "f"(frag_D[1 + ilp * 4]), "f"(frag_D[2 + ilp * 4]), "f"(frag_D[3 + ilp * 4])
                            );
                }
                pattern = brick_patterns[1];
                pattern_first = pattern & 0xffffffff;
                pattern_second = (pattern >> 32) & 0xffffffff;
                index1 = BitCount(pattern_first & pre_bits_mask);
                num_nnz_first = ((counter_pattern >> 16)&0x000000ff);
                index2 = BitCount(pattern_second & pre_bits_mask) + num_nnz_first;
                val1 = ((pattern_first >> lane_id) & 0b1) ? nnzs[count + index1] : 0.0f;
                val2 = ((pattern_second >> lane_id) & 0b1) ? nnzs[count + index2] : 0.0f;
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(val1));
                asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(val2));
                A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
                #pragma unroll
                for(unsigned int ilp = 0; ilp<ILP_LEVEL; ++ilp){
                    asm volatile(
                            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                            : "=f"(frag_D[ILP_LEVEL * 4 + 0 + ilp * 4]), "=f"(frag_D[ILP_LEVEL * 4 + 1 + ilp * 4]), "=f"(frag_D[ILP_LEVEL * 4 + 2 + ilp * 4]), "=f"(frag_D[ILP_LEVEL * 4 + 3 + ilp * 4])
                            : "r"(A[2]), "r"(A[3]),
                    "r"(B[ilp]),
                    "f"(frag_D[ILP_LEVEL * 4 + 0 + ilp * 4]), "f"(frag_D[ILP_LEVEL * 4 + 1 + ilp * 4]), "f"(frag_D[ILP_LEVEL * 4 + 2 + ilp * 4]), "f"(frag_D[ILP_LEVEL * 4 + 3 + ilp * 4])
                            );
                }
                break;
        }
    #endif
}
template <unsigned int NUM_WARPS, unsigned int ILP_LEVEL>
__global__ void SpMM(unsigned int * MatA, float * MatB, float * MatC, uint4 *row_panel_index_array,
                     unsigned int num_row_panels, unsigned int N, unsigned int n_offset, unsigned int *active_cols_vec,
                     unsigned long long int *sizePtr){
    __shared__ float MatA_SEM[2*BLK_MEM_SIZE];
    __shared__ float MatB_SEM[2*TK*(NUM_WARPS * ILP_LEVEL * inst_n + PADDING)];
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t group_id = lane_id >> 2;
    uint32_t tid_in_group = lane_id % 4;
    uint32_t frag_A[4]; // 16 * 16  / 32 = 8 * bf16
    uint32_t frag_B[ILP_LEVEL]; // 8 * 16  / 32
    unsigned int copy_iters;
    unsigned int pre_bits_mask = 0xffffffff >>(32 - lane_id);
    unsigned int n_start = blockIdx.y * (NUM_WARPS * inst_n * ILP_LEVEL) + n_offset;
    uint4 row_panel_info = row_panel_index_array[blockIdx.x];
    uint32_t start = row_panel_info.x;
    uint32_t end = row_panel_info.y;
    unsigned int row_panel_index = row_panel_info.z;
    unsigned int atomic = row_panel_info.w;
    float frag_D[4 * r_tiles * ILP_LEVEL] = {0.0f};
    unsigned long long int offset = sizePtr[start];
    unsigned int * cols = &active_cols_vec[start * TK];
    unsigned int addr;
    #pragma unroll
    for(unsigned int i=threadIdx.x; i<TK*((NUM_WARPS * inst_n * ILP_LEVEL)/4); i+=(NUM_WARPS * 32)){
        unsigned int col = i/((NUM_WARPS * inst_n * ILP_LEVEL)/4);
        unsigned int n = i%((NUM_WARPS * inst_n * ILP_LEVEL)/4);
        addr = __cvta_generic_to_shared(&MatB_SEM[col * (NUM_WARPS * inst_n * ILP_LEVEL + PADDING) + n * 4]);
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(addr), "l"(&MatB[cols[col] * N + n * 4 + n_start]));
    }
    addr = __cvta_generic_to_shared(MatA_SEM);
    #pragma unroll
    for(unsigned int i = warp_id; i < 1; i += NUM_WARPS){
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(addr + i * 512 + lane_id * 16), "l"(&MatA[offset + i * 128 + lane_id * 4]));
    }
    copy_iters = MatA[offset + FETCH_ITERS_OFFSET];
    for(unsigned int i = 1 + warp_id; i<copy_iters; i += NUM_WARPS){
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(addr + i * 512 + lane_id * 16), "l"(&MatA[offset + i * 128 + lane_id * 4]));
    }
    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    uint32_t comp_idx = 0;
    uint32_t load_idx = 0;
    unsigned int * metaData_SEM;
    for(unsigned int block_iter = 1; block_iter < (end - start); ++block_iter){
        load_idx = (block_iter & 1);
        comp_idx = ((block_iter - 1) & 1);
        metaData_SEM = reinterpret_cast<unsigned int *>(&MatA_SEM[comp_idx * BLK_MEM_SIZE]);
        cols = &metaData_SEM[NEXT_COLS_OFFSET];
        unsigned long long int *patterns = reinterpret_cast<unsigned long long int *>(&metaData_SEM[BRICK_PATTERN_OFFSET]);
        float *nnz_array = &MatA_SEM[comp_idx * BLK_MEM_SIZE + MetaDataSize];
        ulonglong4 nested_data = (reinterpret_cast<ulonglong4 *>(&MatA_SEM[comp_idx * BLK_MEM_SIZE]))[0];
        offset = nested_data.x;
        unsigned long long int col_ptr_patterns = nested_data.y;
        #pragma unroll
        for(unsigned int i = threadIdx.x; i < TK*((NUM_WARPS * inst_n * ILP_LEVEL)/4); i+=(NUM_WARPS * 32)){
            unsigned int col = i/((NUM_WARPS * inst_n * ILP_LEVEL)/4);
            unsigned int n = i%((NUM_WARPS * inst_n * ILP_LEVEL)/4);
            addr = __cvta_generic_to_shared(&MatB_SEM[col * (NUM_WARPS * inst_n * ILP_LEVEL + PADDING) + n * 4
                                                      + load_idx * TK * (NUM_WARPS * inst_n * ILP_LEVEL + PADDING)]);
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(addr), "l"(&MatB[cols[col] * N + n * 4 + n_start]));
        }
        addr = __cvta_generic_to_shared(&MatA_SEM[load_idx * BLK_MEM_SIZE]);
        #pragma unroll
        for(unsigned int i = warp_id; i < 1; i += NUM_WARPS){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(addr + i * 512 + lane_id * 16), "l"(&MatA[offset + i * 128 + lane_id * 4]));
        }
        ulonglong2 nested_data2 = (reinterpret_cast<ulonglong2 *>(&MatA_SEM[comp_idx * BLK_MEM_SIZE + NNZ_COUNTER_OFFSET]))[0];
        copy_iters = (reinterpret_cast<unsigned int *>(&nested_data2.y))[1];
        for(unsigned int i = 1 + warp_id; i<copy_iters; i += NUM_WARPS){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(addr + i * 512 + lane_id * 16), "l"(&MatA[offset + i * 128 + lane_id * 4]));
        }
        ushort *nnz_counters = reinterpret_cast<ushort *>(&nested_data2.x);
        unsigned int * brick_num_nnz_counter1_arr = reinterpret_cast<unsigned int *>(&nested_data.z);
        unsigned int * brick_num_nnz_counter2_arr = reinterpret_cast<unsigned int *>(&nested_data.w);
        #pragma unroll
        for(unsigned int i=0; i<2; i++){
            float * brick_b_address = &MatB_SEM[comp_idx * TK * (NUM_WARPS * inst_n * ILP_LEVEL + PADDING)
                                   + (tid_in_group + i * inst_k) * (NUM_WARPS * inst_n * ILP_LEVEL + PADDING) + warp_id * (inst_n * ILP_LEVEL) + group_id];
            unsigned int col_ptr_pattern = ((col_ptr_patterns >>(i * 4))&0b1111);
            unsigned int count = nnz_counters[i];
            COMPUTE<ILP_LEVEL>(col_ptr_pattern, nnz_array, pre_bits_mask, count,
                               &patterns[i * r_tiles], lane_id, frag_A, frag_B, frag_D, brick_b_address, brick_num_nnz_counter1_arr[i]);
        }
        #pragma unroll
        for(unsigned int i=2; i<4; i++){
            float * brick_b_address = &MatB_SEM[comp_idx * TK * (NUM_WARPS * inst_n * ILP_LEVEL + PADDING)
                                                + (tid_in_group + i * inst_k) * (NUM_WARPS * inst_n * ILP_LEVEL + PADDING) + warp_id * (inst_n * ILP_LEVEL) + group_id];
            unsigned int col_ptr_pattern = ((col_ptr_patterns >>(i * 4))&0b1111);
            unsigned int count = nnz_counters[i];
            COMPUTE<ILP_LEVEL>(col_ptr_pattern, nnz_array, pre_bits_mask, count,
                               &patterns[i * r_tiles], lane_id, frag_A, frag_B, frag_D, brick_b_address, brick_num_nnz_counter2_arr[i - 2]);
        }
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
    metaData_SEM = reinterpret_cast<unsigned int *>(&MatA_SEM[load_idx * BLK_MEM_SIZE]);
    unsigned long long int *patterns = reinterpret_cast<unsigned long long int *>(&metaData_SEM[BRICK_PATTERN_OFFSET]);
    float *nnz_array = &MatA_SEM[load_idx * BLK_MEM_SIZE + MetaDataSize];
    ulonglong4 nested_data = (reinterpret_cast<ulonglong4 *>(&MatA_SEM[load_idx * BLK_MEM_SIZE]))[0];
    ulonglong2 nested_data2 = (reinterpret_cast<ulonglong2 *>(&MatA_SEM[load_idx * BLK_MEM_SIZE + NNZ_COUNTER_OFFSET]))[0];
    unsigned long long int col_ptr_patterns = nested_data.y;
    ushort *nnz_counters = reinterpret_cast<ushort *>(&nested_data2.x);
    unsigned int * brick_num_nnz_counter1_arr = reinterpret_cast<unsigned int *>(&nested_data.z);
    unsigned int * brick_num_nnz_counter2_arr = reinterpret_cast<unsigned int *>(&nested_data.w);
    #pragma unroll
    for(unsigned int i=0; i<2; i++){
        unsigned int col_ptr_pattern = ((col_ptr_patterns >>(i * 4))&0b1111);
        unsigned int count = nnz_counters[i];
        if(col_ptr_pattern == 0){
            break;
        }
        float * brick_b_address = &MatB_SEM[load_idx * TK * (NUM_WARPS * inst_n * ILP_LEVEL + PADDING)
                                            + (tid_in_group + i * inst_k) * (NUM_WARPS * inst_n * ILP_LEVEL + PADDING) + warp_id * (inst_n * ILP_LEVEL) + group_id];
        COMPUTE<ILP_LEVEL>(col_ptr_pattern, nnz_array, pre_bits_mask, count,
                           &patterns[i * r_tiles], lane_id, frag_A, frag_B, frag_D,brick_b_address, brick_num_nnz_counter1_arr[i]);
    }
    #pragma unroll
    for(unsigned int i=2; i<4; i++){
        unsigned int col_ptr_pattern = ((col_ptr_patterns >>(i * 4))&0b1111);
        unsigned int count = nnz_counters[i];
        if(col_ptr_pattern == 0){
            break;
        }
        float * brick_b_address = &MatB_SEM[load_idx * TK * (NUM_WARPS * inst_n * ILP_LEVEL + PADDING)
                                            + (tid_in_group + i * inst_k) * (NUM_WARPS * inst_n * ILP_LEVEL + PADDING) + warp_id * (inst_n * ILP_LEVEL) + group_id];
        COMPUTE<ILP_LEVEL>(col_ptr_pattern, nnz_array, pre_bits_mask, count,
                           &patterns[i * r_tiles], lane_id, frag_A, frag_B, frag_D,brick_b_address, brick_num_nnz_counter2_arr[i - 2]);
    }
    float2 *frag_D2 = reinterpret_cast<float2 *>(frag_D);
    if(atomic){
        #pragma unroll
        for(uint32_t i = 0;i<r_tiles;++i){
            #pragma unroll
            for(unsigned int ii=0; ii<ILP_LEVEL; ++ii){
                #pragma unroll
                for(int j = 0; j < 4; j++){
                    uint32_t row_d = (j<2)?group_id:group_id+8;
                    uint32_t col_d = (tid_in_group * 2) + (j & 0x1);
                    atomicAdd(&MatC[(row_d + row_panel_index * TM + i*inst_m)*N + warp_id * inst_n * ILP_LEVEL + ii * inst_n + col_d + n_start],
                              frag_D[i * ILP_LEVEL * 4 + ii * 4 + j]);
                }
            }
        }
    }else{
        #pragma unroll
        for(uint32_t i = 0;i < r_tiles; ++i){
            #pragma unroll
            for(unsigned int ii=0; ii<ILP_LEVEL; ++ii){
                #pragma unroll
                for(unsigned int j=0; j < 2; ++j){
                    uint32_t row_d = (j == 0)?group_id:group_id+8;
                    uint32_t col_d = tid_in_group * 2;
                    float2 *global_buffer = reinterpret_cast<float2 *>(&MatC[(row_panel_index * TM + i * inst_m + row_d)*N
                                                                             + warp_id * inst_n * ILP_LEVEL + ii * inst_n + n_start + col_d]);
                    global_buffer[0] = frag_D2[i * ILP_LEVEL * 2 + ii * 2 + j];
                }
            }
        }
    }
}
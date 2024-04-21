//
// Created by lizhi on 1/10/24.
//

#ifndef PATTERN_MERGE_INC_H
#define PATTERN_MERGE_INC_H
#include <mma.h>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <stdio.h>
#define inst_m 16
#define inst_n 8
#define inst_k 4
#ifndef TM
#define TM 16
#endif
#ifndef TK
#define TK 16
#endif
#define r_tiles (TM/inst_m)
#define NUM_TM_BRICKS ((TM - 1)/inst_m + 1)
#define NUM_TK_BRICKS ((TK - 1)/inst_k + 1)
#define NEXT_BLK_MEM_ADDR_OFFSET 0
#define COL_PTR_OFFSET 2
#define PATTERN_NNZ_COUNTER_OFFSET 4
#define NEXT_COLS_OFFSET 8
#define NNZ_COUNTER_OFFSET (TK + 8)
#define FETCH_ITERS_OFFSET (TK + 10)
#define NEXT_FETCH_ITERS_OFFSET (TK + 11)
#define BRICK_PATTERN_OFFSET (TK + 12)
#define MetaDataSize (TK + NUM_TM_BRICKS * NUM_TK_BRICKS * 2 + 12)
#define BLK_MEM_SIZE (TM * TK + 128)
#ifndef Machine
#define Machine 0
#endif
using COMPUTETYPE = float;
class BrickPos{
public:
    unsigned int row;
    unsigned int col;
    BrickPos(unsigned int row_, unsigned int col_){
        row = row_;
        col = col_;
    }
};
class COO{
public:
    unsigned int r;
    unsigned int c;
    float val;
    COO(unsigned int r_, unsigned int c_, float val_){
        r = r_;
        c = c_;
        val = val_;
    }
};

inline void chkerr(cudaError_t code, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) <<" at line "<<line<<std::endl;
        exit(-1);
    }
}
void spmm_coo(unsigned int *rows, unsigned int *cols, float * vals, float * B,
              float * C, uint32_t M, uint32_t K,uint32_t N, unsigned int NNZ){
    memset(C, 0, M*N*sizeof(float));
    for(uint32_t i=0;i<NNZ;++i){
        uint32_t r = rows[i];
        uint32_t c = cols[i];
        float v = vals[i];
#pragma omp parallel for
        for(uint32_t j=0;j<N;++j){
            C[r*N+j] += v*B[c*N+j];
        }
    }
}
float check_gemm_correctness_coo(float * C, float * C_HOST, uint32_t M, uint32_t N){
    float diff = 0.0f;
#pragma omp parallel
    {
        float diff_local = 0.0f;
#pragma omp for nowait
        for(unsigned int i = 0;i<M*N; ++i){
            if(C[i] == 0.0f)
                continue;
            if(abs(C[i] - C_HOST[i])/abs(C[i]) > diff_local){
                if(C[i] != 0.0f){
                    diff_local = abs(C[i] - C_HOST[i])/abs(C[i]);
                }else{
                    diff_local = abs(C[i] - C_HOST[i]);
                }
            }
        }
#pragma omp critical
        {
            if(diff_local>diff) {
                diff = diff_local;
            }
        }
    }
    return diff;
}
class Pos{
public:
    unsigned int r;
    unsigned int c;
    Pos(unsigned int r_, unsigned int c_){
        r = r_;
        c = c_;
    }
};
struct cmpPosCSR {
    bool operator()(const Pos & a, const Pos & b) const {
        if(a.r < b.r){
            return true;
        }else if (a.r == b.r){
            return a.c < b.c;
        }else{
            return false;
        }
    }
};
struct cmpPosCSC {
    bool operator()(const BrickPos & a, const BrickPos & b) const {
        if(a.col < b.col){
            return true;
        }else if (a.col == b.col){
            return a.row < b.row ;
        }else{
            return false;
        }
    }
};
struct cmpCSC {
    bool operator()(const Pos & a, const Pos & b) const {
        if(a.c < b.c){
            return true;
        }else if (a.c == b.c){
            return a.r < b.r ;
        }else{
            return false;
        }
    }
};
struct cmpPosR {
    bool operator()(const Pos & a, const Pos & b) const {
        if(a.r < b.r){
            return true;
        }else{
            return false;
        }
    }
};
#endif //PATTERN_MERGE_INC_H

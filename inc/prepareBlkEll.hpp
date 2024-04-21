#ifndef PREPAREBLKELL_HPP
#define PREPAREBLKELL_HPP
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_set>
#include "SparseCoo.hpp"
#include "CSR.hpp"
#include <cassert>

using namespace std;
#define densePowerFactor 1

template<class DType>
class preprocess
{
    private:
    //// Genneral info
    CSR_Matrix<DType> * A_CSR;
    int BlockSize;

    //// Blocked-ELL dense blocks info
    int* ellColIdx;
    DType * ellValues;
    uint32_t ellColIdx_height;
    uint32_t ellColIdx_width;
    int num_blocks;
    int num_elems;
    /*
     * Explores the A_CSR and computes: ellColIdx_height, ellColIdx_width, and ellColIdx
     */
    void generate_ellColIdx()
    {
        cout<<"1st pass: generate ELLBlk ellColIdx...\n";
        unordered_map<int, unordered_map<int,int>> map;
        //// Go over A_CSR (1st pass) and fill the hash table this->map
        int NNZ_processed = 0;
        for (int row = 0; row < A_CSR->nrows; row++)
        {
            int blockRowIdx = row / BlockSize;
            if(map.find(blockRowIdx) == map.end())
            {
                unordered_map<int, int> cols_map;
                map[blockRowIdx] = cols_map; // generate an empty row panel
            }
            int start = A_CSR->rowPtr[row];
            int end   = A_CSR->rowPtr[row + 1];
            for (int elemIdx = start; elemIdx < end; elemIdx++)
            {
                int col = A_CSR->cols[elemIdx];
                int blockColIdx = col / BlockSize;
                if(map[blockRowIdx].find(blockColIdx) == map[blockRowIdx].end()) // if block does not exist in the row-panel
                {
                    map[blockRowIdx][blockColIdx] = 1;
                }
                else
                {
                    map[blockRowIdx][blockColIdx] ++; // increment the nnz
                }
                NNZ_processed++;
            }
        }
        if(NNZ_processed!=A_CSR->nnz)
        {
            cout<<"===================================\n";
            cout<<"Error At File:"<<__FILE__<<" line:"<<__LINE__<<endl;
            cout<<"NNZ before generating ELLBlk(1st pass):"<<(A_CSR->nnz)<<endl;
            cout<<"NNZ after  generating ELLBlk(1st pass):"<<NNZ_processed<<endl;
            assert(NNZ_processed==A_CSR->nnz && "Error: nnz before and after generating ELLBlk (1st pass) does not match.");
        }
        ellColIdx_height = map.size();

        //// find the row-panel with max number of dense blocks and set ellColIdx_width to the max val.
        ellColIdx_width = 0;
        for(auto &_rpanel : map) 
        {
            int numblks = _rpanel.second.size();
            if (numblks > ellColIdx_width)
                ellColIdx_width = numblks;
        }
        //// Go over the hash table this->map and generate ellColIdx
        //// count the number of diff block types
        int ellColIdx_size = ellColIdx_height * ellColIdx_width;
        ellColIdx = new int[ellColIdx_size];
        memset (ellColIdx, -1, sizeof (int) * (ellColIdx_size)); // initiate ellColIdx to -1
        
        #pragma omp parallel for
        for(auto &_rpanel : map)
        {
            int blockRowIdx = _rpanel.first;
            int dense_blk_index = 0;
            //// sort the row-panel based on the blockColIdx
            auto rpanel_sorted_blks = vector<pair<int, int>>(begin(_rpanel.second), end(_rpanel.second));
            std::sort(begin(rpanel_sorted_blks), end(rpanel_sorted_blks));
            
            for (auto &_blk : rpanel_sorted_blks)
            {
                int blockColIdx = _blk.first;
                num_blocks++;
                //// update the  ellColIdx 
                int offset = blockRowIdx * ellColIdx_width + dense_blk_index;
                if(offset >= ellColIdx_size)
                {
                    cout<<"blockRowIdx:"<<blockRowIdx<<endl;
                    cout<<"dense_blk_index:"<<dense_blk_index<<endl;
                    cout<<"index accessed:"<<offset<<endl;
                    cout<<"ellColIdx_size:"<<ellColIdx_size<<endl;
                    cout<<"Memory Error at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                    cout<<"Error !!\n";
                    assert(offset < ellColIdx_size && "analysis: out of bounday acccess!!");
                }
                ellColIdx[offset] = blockColIdx;
                dense_blk_index++;
            }
        }
    }

    /*
     * Explores this->ellColIdx and generates the ellValues 
     */
    void generate_ellValues()
    {
        cout<<"2nd pass: generate ELLBlk ellValues...\n";
        uint64_t ellColIdx_size = 0;
        ellColIdx_size = (uint32_t) ellColIdx_height * ellColIdx_width;
        uint64_t ellValues_size = ellColIdx_size * BlockSize * BlockSize;
        ellValues = new DType[ellValues_size];
        memset (ellValues, (DType)0., sizeof (DType) * (ellValues_size)); // initiate ellValues to 0
        //// Go over all A_CSR (2nd pass)
        // traverse each non-zero and 
        // - assign it either to a dense block in Blocked-ELL 
        #pragma omp parallel for
        for (int row = 0; row < A_CSR->nrows; row++) 
        {
            int start = A_CSR->rowPtr[row];
            int end = A_CSR->rowPtr[row + 1];
            for (int elemIdx = start; elemIdx < end; elemIdx++)
            {
                int col = A_CSR->cols[elemIdx];
                int blockRowIdx = row / BlockSize;
                int blockColIdx = col / BlockSize;
                int EIdx = find_EIdx(blockRowIdx, blockColIdx); // -1 if the row-panel does not own a block with BlockColIdx
                DType val = (DType) A_CSR->values[elemIdx]; // type cating
                uint32_t dest_idx = ellColIdx_width * BlockSize * row + EIdx * BlockSize + col%BlockSize;
                if(dest_idx >= ellValues_size )
                {
                    cout<<"row:"<<row<<" col:"<<col<<endl<<"BlockSize:"<<BlockSize<<endl<<"ellColWidth:"<<ellColIdx_width<<endl<<"ellColIdx_height:"<<ellColIdx_height<<endl; 
                    cout<<"dest_idx:"<<dest_idx<<endl<<"ellValues_size:"<<ellValues_size<<endl;
                    cout<<"Memory Error at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                    cout<<"Error !!\n";
                    assert (false && "copying dense block... out of boundary access!!\n");
                }
                ellValues[dest_idx] = val; // copy the element value to the destination dense block(also do the type casting)
                num_elems++;
            }
        }
    }
    int find_EIdx(int blockRowIdx, int blockColIdx)
    {
        int begin = blockRowIdx * ellColIdx_width;
        if (ellColIdx[begin] == -1)
            return -1; // empty row

        int end = (blockRowIdx + 1) * ellColIdx_width - 1;
        while (ellColIdx[end] == -1 && end >= begin)
            end--;

        if (ellColIdx[begin] > blockColIdx)
            return -1; // not found

        while (begin <= end)
        {
            int mid = begin + (end - begin) / 2;
            if (ellColIdx[mid] == blockColIdx)
                return mid % ellColIdx_width;
            else if (ellColIdx[mid] > blockColIdx)
                end = mid - 1;
            else
                begin = mid + 1;
        }
        return -1; // not found
    }
    
    public:
    preprocess(CSR_Matrix<DType> * A_CSR, int BlockSize)
    {
        this->A_CSR = A_CSR;
        this->BlockSize = BlockSize;
        ellColIdx_height = 0;
        ellColIdx_width = 0;
        num_elems = 0;
        num_blocks = 0;
        generate_ellColIdx();
        generate_ellValues();
    }
    ~preprocess()
    {
        delete[] ellColIdx;
        delete[] ellValues;
    }
    void print_ellColIdx()
    {
        cout<<"\n===========================\n";
        cout<<"Blocked-Ell ellColIdx matrix:\n";
        cout<<"ellColIdx_height:"<<ellColIdx_height<<"\tellColIdx_width:"<<ellColIdx_width<<endl<<endl;
        for(int rowBlk = 0; rowBlk < ellColIdx_height; rowBlk++)
        {
            int colBlk = 0;
            for(colBlk = 0; colBlk < ellColIdx_width; colBlk++)
            {
                cout<<ellColIdx[rowBlk * ellColIdx_width + colBlk];
                if(colBlk != ellColIdx_width)
                    cout<<",";
            }
            if(colBlk>0)
                cout<<endl;
        }
        cout<<"\n===========================\n";
    }
    void print_ellValues()
    {
        cout<<"\n===========================\n";
        cout<<"Blocked-Ell ellValues matrix:\n";
        cout<<"BlockSize:"<<BlockSize<<"\tellColIdx_width:"<<ellColIdx_width<<endl<<endl;
        cout<<"height:"<<ellColIdx_height * BlockSize<<"\twidth:"<<ellColIdx_width * BlockSize<<endl<<endl;
        for(int row = 0; row < ellColIdx_height * BlockSize; row++)
        {
            int col = 0;
            for(col = 0; col < ellColIdx_width*BlockSize; col++)
            {
                cout<<(float)ellValues[row * ellColIdx_width*BlockSize + col];
                if(col != ellColIdx_width*BlockSize - 1)
                    cout<<",";
            }
            if(col>0)
                cout<<endl;
        }
        cout<<"\n===========================\n";
    }

    void blockedELLToDense(DType* dense)
    {
        for(int bRow = 0; bRow < ellColIdx_height; bRow++)
        {
            for(int bCol = 0; bCol < ellColIdx_width; bCol++)
            {
                int blk_actual_Col = ellColIdx[bRow * ellColIdx_width + bCol];
                if(blk_actual_Col >= 0) // not a placeholder block
                {
                    // copy the block from ellValues[bRow][bCol] into the dense[bRow][blk_actual_Col]
                    int blk_start_row = bRow * BlockSize;
                    int blk_end_row = (bRow + 1) * BlockSize;
                    for(int r = blk_start_row; r < blk_end_row; r++)
                    {
                        for(int c = 0; c < BlockSize; c++)
                        {
                            int src_col  = c + bCol * BlockSize;
                            int dest_col = c + blk_actual_Col * BlockSize;
                            dense[r * (A_CSR->ncols) + dest_col ] = ellValues[r * (ellColIdx_width * BlockSize) + src_col];
                        }
                    }
                }
            }
        }
    }
    int* get_ellColIdx()
    {
        return ellColIdx;
    }
    int get_ellColIdx_height()
    {
        return ellColIdx_height;
    }
    int get_ellColIdx_width()
    {
        return ellColIdx_width;
    }
    DType * get_ellValues()
    {
        return ellValues;
    }
    int get_num_blocks()
    {
        return num_blocks;
    }
    int get_num_elems()
    {
        return num_elems;
    }
};
#endif
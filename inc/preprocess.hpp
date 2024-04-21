#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_set>
#include "SparseCoo.hpp"
#include "CSR.hpp"
#include <cassert>

using namespace std;
#define DType __half
#define densePowerFactor 1

class preprocess
{
    private:
    //// Genneral info
    CSR_Matrix * A_CSR;
    int BlockSize;
    float deviation_threshold;
    float underloadDominance;
    int total_blocks; // total number of sparse and dense blocks

    //// Blocked-ELL dense blocks info
    int* ellColIdx;
    DType * ellValues;
    int ellColIdx_height;
    int ellColIdx_width;
    int num_dense_blocks;
    int num_dense_elems;

    //// 2:4 sparse blocks info (SP24)
    //// key: block offset, value: <colIdxs,values> 
    unordered_map<int, pair<int*, DType*>>* SP24Blocks;
    DType* SP24_vals;           // blocks' vals
    uint32_t* SP24_colIdxs;     // blocks' colIdx
    int * SP24_offsets;         // blocks' offsets
    int num_24Sparse_blocks;
    int num_24Sparse_elems;
    //// Blocked-ELL dense format of sparse blocks for comparison purposes
        int* SP24_ellColIdx;
        DType * SP24_ellValues;
        int SP24_ellColIdx_height;
        int SP24_ellColIdx_width;
    //// ===========================

    //// Ordinary sparse blocks (COO)
    int num_VerySparse_elems_total; // total number of verysparse elements
    int num_VerySparse_elems;       // number of very sparse elements comming from the verysparse blocks
    int num_VerySparse_elems_split; // number of very sparse elements that are remnants of a split from another block to guarantee it is 2:4
    int num_VerySparse_blocks;
    SparseCoo* COOBlksCombined;
    unordered_map<int, unordered_set<int>> verySparseBlks; // keeps the (rowBlk, colBlk) of the verySparseBlocks
    /*
     * Explores the A_CSR and generates and fills the hash tabel this.map
     * - The key in the hash table is "row panel ID"
     * - The value is another hash table in which 
        - the key is "column block ID"
        - the value is vactor of 3 elements [NNZ, # underloaded bands, # of overloaded bands]
     * Each block is BlockSize x BlockSize
     */
    void analysis()
    {
        cout<<"1st pass: Analyzing of the A_CSR matrix...\n";
        unordered_map<int, unordered_map<int,vector<int>>> map;
        //// Go over A_CSR (1st pass) and fill the hash table this->map
        for (int row = 0; row < A_CSR->nrows; row++)
        {
            int blockRowIdx = row / BlockSize;
            if(map.find(blockRowIdx) == map.end())
            {
                unordered_map<int, vector<int>> cols_map;
                map[blockRowIdx] = cols_map; // generate an empty row panel
            }
            int start = A_CSR->rowPtr[row];
            int end = A_CSR->rowPtr[row + 1];
            int previousBandIdx  = 0;
            int previousBandNNZ = 0;
            for (int elemIdx = start; elemIdx < end; elemIdx++)
            {
                int col = A_CSR->cols[elemIdx];
                int blockColIdx = col / BlockSize;
                int currentBandIdx = col / 4;
                if(previousBandIdx != currentBandIdx) // if bandIdx changed
                {
                    //// 1. update the type of previuos band: increment undeloads or overloads as needed
                    int pre_band_colBlk = previousBandIdx / (BlockSize/4); // Block column that previous band belong to
                    if(previousBandNNZ < 2) // underload band
                    {
                        if(map[blockRowIdx].find(pre_band_colBlk) == map[blockRowIdx].end())
                        {
                            vector<int> blkInfo {0,1,0}; // nnz = 0, underloads = 1, overloads = 0
                            map[blockRowIdx][pre_band_colBlk] = blkInfo;
                        }
                        else
                            map[blockRowIdx][pre_band_colBlk][1] = map[blockRowIdx][pre_band_colBlk][1] + 1;
                    }
                    else if(previousBandNNZ > 2) // overload
                    {
                        if(map[blockRowIdx].find(pre_band_colBlk) == map[blockRowIdx].end())
                        {
                            vector<int> blkInfo {0,0,1}; // nnz = 0, underloads = 0, overloads = 1
                            map[blockRowIdx][pre_band_colBlk] = blkInfo;
                        }
                        else
                            map[blockRowIdx][pre_band_colBlk][2] = map[blockRowIdx][pre_band_colBlk][2] + 1;
                    }
                    //// 2. take care of the skipped empty bands
                    for(int bi = previousBandIdx + 1; bi < currentBandIdx; bi++) // count the skipped empty bands and tie them to their blocks
                    {
                        int band_colBlk = bi / (BlockSize/4); // Block column that previous band belong to
                        if(map[blockRowIdx].find(band_colBlk) == map[blockRowIdx].end()) // if block does not exist in the row-panel
                        {
                            vector<int> blkInfo {0,1,0}; // nnz = 0, underloads = 1, overloads = 0
                            map[blockRowIdx][band_colBlk] = blkInfo;
                        }
                        else
                        {
                            map[blockRowIdx][band_colBlk][1]  = map[blockRowIdx][band_colBlk][1] + 1; // increment the underloads
                        }
                    }
                    //// 3. take care of the current band
                    previousBandIdx = currentBandIdx;     // reset the previous band idx and nnz
                    previousBandNNZ = 1;
                    if(map[blockRowIdx].find(blockColIdx) == map[blockRowIdx].end()) // if block does not exist in the row-panel
                    {
                        vector<int> blkInfo {1,0,0}; // nnz = 1, underloads = 0, overloads = 0
                        map[blockRowIdx][blockColIdx] = blkInfo;
                    }
                    else
                    {
                        map[blockRowIdx][blockColIdx][0] = map[blockRowIdx][blockColIdx][0] + 1; // increment the nnz
                    }
                }
                else // still in the same band
                {
                    previousBandNNZ ++;
                    if(map[blockRowIdx].find(blockColIdx) == map[blockRowIdx].end()) // if block does not exist in the row-panel
                    {
                        vector<int> blkInfo {1,0,0}; // nnz = 1, underloads = 0, overloads = 0
                        map[blockRowIdx][blockColIdx] = blkInfo;
                    }
                    else
                    {
                        map[blockRowIdx][blockColIdx][0] = map[blockRowIdx][blockColIdx][0] + 1; // increment the nnz
                    }
                }
            }
        }
        unordered_map<int, unordered_map<int,vector<int>>> mapNonEmpty,t;

        for(auto &_rpanel : map) // just keep the non-empty blocks
        {
            unordered_map<int, vector<int>> cols_map;
            mapNonEmpty[_rpanel.first] = cols_map; // generate an empty row panel
            for(auto &_blk :_rpanel.second)
            {
                if(_blk.second[0] != 0 )
                {
                    mapNonEmpty[_rpanel.first][_blk.first] = _blk.second;
                    total_blocks++;
                }
            }
        }
        map  = t;
        ellColIdx_height = mapNonEmpty.size();
        SP24_ellColIdx_height = mapNonEmpty.size(); // collected only for comparison purpose
        //// find the row-panel with max number of dense blocks and set ellColIdx_width to the max val.
        //// a block is dense if  (#underloads + #overloads) / ((BS * BS) / 4) > DeviationThreshold && overloads >= underloads
        for(auto &_rpanel : mapNonEmpty) 
        {
            int num_dense_blks = 0;
            int num_verySparse_blks = 0;
            int num_SP24_blks = 0; // collected only for comparison purpose
            for(auto &_blk :_rpanel.second)
            {
                float distanceFromPerfect = 4 * ((float)_blk.second[1] + (float)_blk.second[2]) / (BlockSize * BlockSize);
                float underloadRatio = (float) _blk.second[1] / (_blk.second[1]+_blk.second[2]) / densePowerFactor; // the ratio of underloads over (underload + overload ) devided by densePowerFactor
                if( distanceFromPerfect > deviation_threshold && underloadRatio <= underloadDominance)
                    num_dense_blks++;
                else if( distanceFromPerfect > deviation_threshold && underloadRatio > underloadDominance)
                    num_verySparse_blks++;
                else
                    num_SP24_blks++;
            }
            if (num_dense_blks > ellColIdx_width)
                ellColIdx_width = num_dense_blks;

            if (num_SP24_blks > SP24_ellColIdx_width)
                SP24_ellColIdx_width = num_SP24_blks;
        }

        //// Go over the hash table this->mapNonEmpty and generate ellColIdx
        //// count the number of diff block types
        int ellColIdx_size = ellColIdx_height * ellColIdx_width;
        ellColIdx = new int[ellColIdx_size];
        memset (ellColIdx, -1, sizeof (int) * (ellColIdx_size)); // initiate ellColIdx to -1
       
        //// collected only for comparison purpose
        //// go over the hash table this->mapNonEmpty and generate sp24_ellColIdx    
            int SP24_ellColIdx_size = SP24_ellColIdx_height * SP24_ellColIdx_width;
            SP24_ellColIdx = new int[SP24_ellColIdx_size];
            memset (SP24_ellColIdx, -1, sizeof (int) * (SP24_ellColIdx_size)); // initiate SP24_ellColIdx to -1
        ////=====================================
        for(auto &_rpanel : mapNonEmpty)
        {
            int blockRowIdx = _rpanel.first;
            int dense_blk_index = 0;
            int sparse_blk_index = 0; // collected only for comparison purpose
            
            //// sort the row-panel based on the blockColIdx
            auto rpanel_sorted_blks = vector<pair<int, vector<int>>>(begin(_rpanel.second), end(_rpanel.second));
            std::sort(begin(rpanel_sorted_blks), end(rpanel_sorted_blks));
            
            for (auto &_blk : rpanel_sorted_blks)
            {
                int blockColIdx = _blk.first;
                float distanceFromPerfect = 4 * ((float)_blk.second[1] + (float)_blk.second[2]) / (BlockSize * BlockSize);
                float underloadRatio =((_blk.second[1]+_blk.second[2]) == 0)? 0: (float) _blk.second[1] / (_blk.second[1]+_blk.second[2]) / densePowerFactor; // the ratio of underloads over (underload + overload ) devided by densePowerFactor
                if( distanceFromPerfect > deviation_threshold && underloadRatio <= underloadDominance)
                {
                    // cout<<underloadRatio<<","<<-log10(underloadRatio)<<endl;
                    num_dense_blocks++;
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
                else if (distanceFromPerfect > deviation_threshold && underloadRatio > underloadDominance)
                {
                    // cout<<underloadRatio<<","<<-log10(underloadRatio)<<endl;
                    num_VerySparse_blocks++;
                    if(verySparseBlks.find(blockRowIdx) == verySparseBlks.end())
                    {
                        unordered_set<int> BlkCols;
                        BlkCols.insert(blockColIdx);
                        verySparseBlks[blockRowIdx] = BlkCols;
                    }
                    else
                    {
                        verySparseBlks[blockRowIdx].insert(blockColIdx);
                    }
                }
                else // 2:4 sparse
                {
                    num_24Sparse_blocks++;
                    //// update the  SP24_ellColIdx 
                    int offset = blockRowIdx * SP24_ellColIdx_width + sparse_blk_index;
                    if(offset >= SP24_ellColIdx_size)
                    {
                        cout<<"blockRowIdx:"<<blockRowIdx<<endl;
                        cout<<"sparse_blk_index:"<<sparse_blk_index<<endl;
                        cout<<"index accessed:"<<offset<<endl;
                        cout<<"SP24_ellColIdx_size:"<<SP24_ellColIdx_size<<endl;
                        cout<<"SP24_ellColIdx_height:"<<SP24_ellColIdx_height<<endl;
                        cout<<"SP24_ellColIdx_width:"<<SP24_ellColIdx_width<<endl;
                        cout<<"Memory Error at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                        cout<<"Error !!\n";
                        assert(offset < SP24_ellColIdx_size && "analysis: out of bounday acccess!!");
                    }
                    SP24_ellColIdx[offset] = blockColIdx;
                    sparse_blk_index++;
                }
            }
        }
    }

    /*
     * Explores this->ellColIdx and generates the ellValues + Sparse blocks for ELL-format and COO format.
     * Sparse blocks that belongs to a very sparse block (distanceFromPerfect > deviation_threshold && _blk.second[2] < _blk.second[1]) are processed as COO
     * Sparse blocks get split in 2 blocks one ELL-format block which holds 2:4 sparsity pattern and one COO format
     */
    void process_blocks()
    {
        cout<<"2nd pass: converting the data from A_CSR matrix to dense and sparse...\n";
        uint64_t ellColIdx_size = (uint64_t) ellColIdx_height * ellColIdx_width;
        uint64_t ellValues_size = ellColIdx_size * BlockSize * BlockSize;
        ellValues = new DType[ellValues_size];
        memset (ellValues, (DType)0., sizeof (DType) * (ellValues_size)); // initiate ellValues to 0
        
        //// collected only for comparison purpose
            uint64_t SP24_ellColIdx_size = (uint64_t) SP24_ellColIdx_height * SP24_ellColIdx_width;
            uint64_t SP24_ellValues_size = SP24_ellColIdx_size * BlockSize * BlockSize;
            SP24_ellValues = new DType[SP24_ellValues_size];
            memset (SP24_ellValues, (DType)0., sizeof (DType) * (SP24_ellValues_size)); // initiate SP24_ellValues to 0
        ////======================================

        //// Go over all A_CSR (2nd pass)
        // traverse each non-zero and 
        // - assign it either to a dense block in Blocked-ELL format or 
        // - if it belongs to a sparse block 
        //   - assign it to a 2:4 Ell-Pack block or 
        //   - assign it to a COO block if it is a excess element (split consept)
        for (int row = 0; row < A_CSR->nrows; row++) 
        {
            int start = A_CSR->rowPtr[row];
            int end = A_CSR->rowPtr[row + 1];

            int current_band_idx = 0; // Index of the current band of 4 consequitive elements
            int band_nnz = 0; // should be <= 2 for a 2:4 band
            for (int elemIdx = start; elemIdx < end; elemIdx++)
            {
                int col = A_CSR->cols[elemIdx];
                int blockRowIdx = row / BlockSize;
                int blockColIdx = col / BlockSize;
                int EIdx = find_EIdx(blockRowIdx, blockColIdx,ellColIdx,0); // -1 if the row-panel does not own a block with BlockColIdx
                if(EIdx >= 0) // if belongs to a dense block
                {
                    DType val = (DType) A_CSR->values[elemIdx];
                    uint32_t dest_idx = ellColIdx_width * BlockSize * row + EIdx * BlockSize + col%BlockSize;
                    if(dest_idx >= ellValues_size )
                    {
                        cout<<"EIdx:"<<EIdx<<endl<<"row:"<<row<<" col:"<<col<<endl<<"BlockSize:"<<BlockSize<<endl<<"ellColWidth:"<<ellColIdx_width<<endl<<"ellColIdx_height:"<<ellColIdx_height<<endl; 
                        cout<<"dest_idx:"<<dest_idx<<endl<<"ellValues_size:"<<ellValues_size<<endl;
                        cout<<"Memory Error at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                        cout<<"Error !!\n";
                        assert (false && "copying dense block... out of boundary access!!\n");
                    }
                    ellValues[dest_idx] = val; // copy the element value to the destination dense block(also do the type casting)
                    num_dense_elems++;
                }
                else if(verySparseBlks.find(blockRowIdx) != verySparseBlks.end() 
                     && verySparseBlks[blockRowIdx].find(blockColIdx) != verySparseBlks[blockRowIdx].end()) // else if the element belongs to a very sparse block just add it to the COO
                {
                    COOBlksCombined->Insert_Next(row, col, A_CSR->values[elemIdx]);
                    num_VerySparse_elems++;
                    num_VerySparse_elems_total++;
                }
                else // if sparse and dense enough then split it to 2:4 and COO
                {
                    //// split the block into COO and 2:4 SP24
                    if(col/4 != current_band_idx) // band changed
                    {
                        current_band_idx = col/4;
                        band_nnz = 0; // reset
                    }
                    // cout<<"row:"<<row<<" col:"<<col<<" ==> ";
                    if(band_nnz < 2) // if the band is not full
                    {
                        insert_next_To_SP24_Blocks(blockRowIdx, blockColIdx, row%BlockSize, col%BlockSize, A_CSR->values[elemIdx]);
                        num_24Sparse_elems++;
                        
                        //// collected only for comparison purpose
                            int EIdx = find_EIdx(blockRowIdx, blockColIdx,SP24_ellColIdx,1); // -1 if the row-panel does not own a block with BlockColIdx
                            // if(EIdx < 0)
                            // {
                            //     cout<<"Error at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                            //     cout<<"Error !!\n";
                            //     assert (EIdx >=0 && "impossible case: sparse block not found in SP24_ellColIdx!! this will affect sp24 passed to DTCU time\n");
                            // }
                            DType val = (DType) A_CSR->values[elemIdx];
                            uint32_t dest_idx = SP24_ellColIdx_width * BlockSize * row + EIdx * BlockSize + col%BlockSize;
                            if(dest_idx >= SP24_ellValues_size )
                            {
                                cout<<"EIdx:"<<EIdx<<endl<<"row:"<<row<<" col:"<<col<<endl<<"BlockSize:"<<BlockSize<<endl<<"ellColWidth:"<<ellColIdx_width<<endl<<"ellColIdx_height:"<<ellColIdx_height<<endl; 
                                cout<<"dest_idx:"<<dest_idx<<endl<<"SP24_ellValues_size:"<<SP24_ellValues_size<<endl;
                                cout<<"Memory Error at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                                cout<<"Error !!\n";
                                assert (false && "copying dense block... out of boundary access!!\n");
                            }
                            SP24_ellValues[dest_idx] = val; // copy the element value to the destination dense block(also do the type casting)
                        ////======================================
                    }
                    else // other
                    {
                        COOBlksCombined->Insert_Next(row, col, A_CSR->values[elemIdx]);
                        num_VerySparse_elems_split++;
                        num_VerySparse_elems_total++;
                    }
                    band_nnz ++;
                }
            }
        }
        // num_24Sparse_blocks= (*SP24Blocks).size();
    }
    int find_EIdx(int blockRowIdx, int blockColIdx, int* ColIdx , bool SP24)
    {
        // binary search the blockRowIdx row-panel for blockColIdx
        int begin = blockRowIdx * (SP24? SP24_ellColIdx_width: ellColIdx_width);
        if(ColIdx[begin] == -1) return -1; // empty row

        int end   = (blockRowIdx + 1) * (SP24? SP24_ellColIdx_width: ellColIdx_width) - 1 ;
        while(ColIdx[end] == -1) end--;
        if ( end == -1) return -1; // empty row
        while(begin <= end)
        {
            int mid = (begin + end)/2;
            if(ColIdx[mid] == blockColIdx) // if found
                return mid % (SP24? SP24_ellColIdx_width: ellColIdx_width); // return the index in the row-panel
            else if (ColIdx[mid] > blockColIdx)
                end = mid - 1;
            else
                begin = mid + 1;
        }
        return -1; // not found
    }
    /*
     * adds element(row, col, val) into block(blockRowIdx,blockColIdx) Note row and col are indices of the element with in the block.
     */
    void insert_next_To_SP24_Blocks(int blockRowIdx,int blockColIdx,int row,int col, DType val)
    {
        // cout<<" insert_next_To_SP24_Blocks,";
        int blk_offset = blockRowIdx * (A_CSR->ncols / BlockSize) + blockColIdx; // block offset in the adj format. BlkRow *(K/BlockSize)+BlkCol
        int blk_compressed_size = BlockSize * BlockSize / 2;
        int bandIdx = col / 4; // col index of the band of 4 consequitive element that this element belongs to with in the block
        int dest_band_offset = bandIdx * 2; // every 4 elements are compressed into 2 elements (compress concept). We are mapping from a 4 elememt band to a 2 element destination band.
        if(SP24Blocks->find(blk_offset) != SP24Blocks->end()) // if block already exists in the list
        {
            // cout<<"r:"<<row<<" c:"<<col<<" val:"<<(float)val<<endl;
            int dest_idx = row * BlockSize/2 + dest_band_offset;
            if ((float)(*SP24Blocks)[blk_offset].second[dest_idx] != 0)// if the 1st element of destination band is already filled then add the element to the 2nd position
                dest_idx++;
            // cout<<"blk_offset:"<<blk_offset<<" dest_idx "<<dest_idx<<endl;
            //// add the element to the block
            (*SP24Blocks)[blk_offset].first [dest_idx] = col%4; // col%4 is the position of element in its source band
            (*SP24Blocks)[blk_offset].second[dest_idx] = val; 
        }
        else
        {
            // cout<<"r:"<<row<<"c:"<<col<<endl;
            //// add the block and then add the element to the block
            int dest_idx = row * BlockSize/2 + dest_band_offset;

            int* colIdx = new int[blk_compressed_size];
            memset (colIdx, 0, sizeof (int) * blk_compressed_size);
            if(dest_idx >= blk_compressed_size)
            {
                cout<<"dest_idx:"<<dest_idx<<" row:"<<row<<" blk_compressed_size:"<<blk_compressed_size<<"dest_band_offset"<<dest_band_offset<<endl;
                cout<<"Memory Error at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                cout<<"Error !!\n";
                assert(false && "out of memory access!!\n");
            }
            // cout<<"blk added, blk_offset :"<<blk_offset<<" dest_idx "<<dest_idx<<endl;
            colIdx[dest_idx] = col%4;

            DType* Vals = new DType[blk_compressed_size];
            memset (Vals, 0., sizeof (DType) * blk_compressed_size);
            Vals[dest_idx] = val;

            (*SP24Blocks)[blk_offset] = {colIdx, Vals};
        }
    }
    public:
    preprocess(CSR_Matrix * A_CSR, int BlockSize,  float deviation_threshold, float underloadDominance = 0.5)
    {
        this->A_CSR = A_CSR;
        this->BlockSize = BlockSize;
        this->deviation_threshold = deviation_threshold;
        this->underloadDominance = underloadDominance;
        total_blocks = 0;
        ellColIdx_height = 0;
        ellColIdx_width = 0;
        SP24_ellColIdx_height = 0;
        SP24_ellColIdx_width = 0;
        SP24Blocks = new unordered_map<int, pair<int*, DType*>>();
        SP24_vals = NULL;
        SP24_colIdxs  = NULL;
        SP24_offsets = NULL;
        COOBlksCombined = new SparseCoo(A_CSR->nrows, A_CSR->ncols, 0);
        num_24Sparse_elems = 0;
        num_VerySparse_blocks = 0;
        num_VerySparse_elems = 0;
        num_VerySparse_elems_split = 0;
        num_VerySparse_elems_total = 0;
        num_dense_elems = 0;
        num_dense_blocks = 0;
        num_24Sparse_blocks = 0;
        cout<<"\n==================================\nPreprocessing...\n";
        analysis();
        process_blocks();
    }
    ~preprocess()
    {
        delete SP24Blocks;

        delete[] ellColIdx;
        delete[] ellValues;

        delete[] SP24_ellColIdx;
        delete[] SP24_ellValues;

        if((*SP24Blocks).size() > 0)
        {
            for(auto &_pair : (*SP24Blocks))
            {
                delete[] _pair.second.first;
                delete[] _pair.second.second;
            }
        }

        if(SP24_vals != NULL)
            delete[] SP24_vals;
        if(SP24_colIdxs  != NULL)
            delete[] SP24_colIdxs;
        if(SP24_offsets != NULL)
            delete[] SP24_offsets;
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
    void print_SP24Values()
    {
        cout<<"====== 2:4 blocks Vals =======\n";
        for(auto &_pair : (*SP24Blocks))
        {
            cout<<"block offset:"<<_pair.first<<endl;
            int BlockWidth = BlockSize / 2; 
            for(int i = 0; i < BlockSize; i++)
            {
                int j;
                for(j = 0; j < BlockWidth; j++)
                {
                    cout<<(float)_pair.second.second[i * BlockWidth + j];
                    if(j != BlockWidth - 1)
                        cout<<",";
                }
                if(j > 0)
                    cout<<endl;
            }
            cout<<"========"<<endl;
        }
    }
    void print_SP24Blocks_ColIdx()
    {
        cout<<"====== 2:4 blocks ColIdx =======\n";
        for(auto &_pair : (*SP24Blocks))
        {
            cout<<"block offset:"<<_pair.first<<endl;
            int BlockWidth = BlockSize / 2; 
            for(int i = 0; i < BlockSize; i++)
            {
                int j;
                for(j = 0; j < BlockWidth; j++)
                {
                    cout<<_pair.second.first[i * BlockWidth + j];
                    if(j != BlockWidth - 1)
                        cout<<",";
                }
                if(j > 0)
                    cout<<endl;
            }
            cout<<"========"<<endl;
        }
    }
    void print_SP24ColIdx()
    {
        for(int blk = 0; blk < get_SP24_size(); blk++)
        {
            for(int i = 0; i < BlockSize * BlockSize/32; i++)
            {
                printf("0x%x",SP24_colIdxs[blk * BlockSize * BlockSize/32 + i]);
                if(i != BlockSize * BlockSize/32-1)
                    cout<<"\n";
            }
            cout<<"\n\n====\n";
        }
    }
    void print_SP24Offsets()
    {
        cout<<"====== 2:4 blocks offsets =======\n";
        for(auto &_pair : (*SP24Blocks))
            cout<<"block offset:"<<_pair.first<<endl;
    }
    void SP24ToDense(DType* dense, int m, int k)
    {
        for(auto &_pair : (*SP24Blocks))
        {
            int blk_offset = _pair.first;
            int blk_offsetY = blk_offset / (A_CSR->ncols/BlockSize);
            int blk_offsetX = blk_offset % (A_CSR->ncols/BlockSize);
            for(int row = 0; row < BlockSize; row++)
            {
                for(int col = 0; col < BlockSize/2; col++)
                {
                    DType val = _pair.second.second[row * (BlockSize/2) + col];
                    if((float)val != 0)
                    {
                        int colIdx = _pair.second.first[row * (BlockSize/2) + col];
                        uint32_t offset = (blk_offsetY * BlockSize + row )* A_CSR->ncols + blk_offsetX * BlockSize + col/2*4+colIdx;
                        if(offset >= m*k)
                        {
                            cout<<"Memory Error at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                            cout<<"Error !!\n";
                            assert(offset < m*k && "out of boundary access in dense[i]");
                        }
                        dense[offset] = val;
                    }
                }
            }
        }
    }
    void blockedELLToDense(DType* dense)
    {
        // int* ellColIdx;
        // DType * ellValues;
        // int ellColIdx_height;
        // int ellColIdx_width;
        // int num_dense_blocks;
        // int num_dense_elems;
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
    /*
     * Converts the BlockedELL Values vector into pointer format
     * Converts the BlockedELL Column indices vector to binary format (2 bit per element)
     */
    void break_SP24Blks_vectors()
    {
        int BlockWidth = BlockSize / 2; // BlockSize / 2 as it is in compressed form
        int numSparseBlks = get_SP24_size();
        SP24_vals = new DType[numSparseBlks * BlockSize * BlockWidth];
        memset (SP24_vals, (DType)0., sizeof (DType) * numSparseBlks * BlockSize * BlockWidth);

        //// 2 bits/element is needed to be able to locate its location in uncompressed 2:4 sparse matrix
        //// M*(K/2) * 2bits/ 32bits = M*K/32 uint32_t
        SP24_colIdxs = new uint32_t[numSparseBlks * (BlockSize * BlockSize / 32)]; // 2 bits per element
        memset (SP24_colIdxs, 0, sizeof (uint32_t)* numSparseBlks * (BlockSize * BlockSize / 32));

        SP24_offsets = new int [numSparseBlks];
        memset (SP24_offsets, 0, sizeof (int) * (numSparseBlks));
        int i = 0;
        int j1 = 0, j2 = 0;
        for( auto &_pair : (*SP24Blocks))
        {
            if(i >= numSparseBlks)
            {
                cout<<"Memory Error at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                cout<<"Error !!\n";
                assert(i < numSparseBlks && "out of boundary access in SP24_offsets[i]");
            }
            //linearize the offsets
            SP24_offsets[i]= _pair.first;

            //linearize the values
            for(int row = 0; row < BlockSize; row++)
            {
                for(int col = 0 ; col < BlockWidth; col++)
                {
                    if(j2 >= numSparseBlks * BlockSize * BlockWidth)
                    {
                        cout<<"Memory Error at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                        cout<<"Error !!\n";
                        assert(j2 < numSparseBlks * BlockSize * BlockWidth && "out of boundary access in SP24_vals[j2]");
                    }
                    SP24_vals[j2] = _pair.second.second[row * BlockWidth + col];
                    j2++;
                }
            }

            //linearize the colIdxs
            for(int row  = 0; row < BlockSize; row++)
            {
                uint32_t element = 0; // each uint32_t element contains 16 2bits colIdx 
                int col;
                for(col = 0; col < BlockWidth; col++)
                {
                    uint32_t colIdx = _pair.second.first[row * BlockWidth + col];
                    // add the 2bits colIdx into the element.MSB position
                    element = (colIdx << 30) | (element >> 2);
                    // printf("%d(%x)",colIdx,element);
                    // if(col != BlockWidth - 1)
                    //     cout<<",";
                    if(col % 16 == 15) // add the element into the array
                    {
                        if(j1 >= (numSparseBlks * (BlockSize * BlockSize / 32)))
                        {
                            cout<<"SP24_offsets[i]:"<<SP24_offsets[i]<<" row:"<<row<<" col:"<<col<<endl;
                            cout<<"index accessed:"<<j1<<" array size:"<<(numSparseBlks * (BlockSize * BlockSize / 32))<<endl;
                            cout<<"Memory Error at file:"<<__FILE__<<" line:"<<__LINE__<<endl;
                            cout<<"Error !!\n";
                            assert(j1 < (numSparseBlks * (BlockSize * BlockSize / 32)) && "out of boundary access in SP24_colIdxs[j1]");
                        }
                        SP24_colIdxs[j1] = element;
                        j1++;
                        element = 0; // reset the number
                    }
                }
                // if(col > 0)
                //     cout<<endl;
            }
            // cout<<"========"<<endl;
            i++;
        }
    }
    int get_total_blocks()
    {
        return total_blocks;
    }
    int* get_ellColIdx()
    {
        return ellColIdx;
    }
    int* SP24_get_ellColIdx()
    {
        return SP24_ellColIdx;
    }
    int get_ellColIdx_height()
    {
        return ellColIdx_height;
    }
    int SP24_get_ellColIdx_height()
    {
        return SP24_ellColIdx_height;
    }
    int get_ellColIdx_width()
    {
        return ellColIdx_width;
    }
    int SP24_get_ellColIdx_width()
    {
        return SP24_ellColIdx_width;
    }
    DType * get_ellValues()
    {
        return ellValues;
    }
    DType * SP24_get_ellValues()
    {
        return SP24_ellValues;
    }
    unordered_map<int, pair<int*, DType*>>* get_SP24Blocks()
    {
        return SP24Blocks;
    }
    SparseCoo* get_COO_Blocks()
    {
        return COOBlksCombined;
    }
    DType* get_SP24_vals()
    {
        return SP24_vals;
    }
    uint32_t* get_SP24_colIdxs()
    {
        return SP24_colIdxs;
    }
    int * get_SP24_offsets()
    {
        return SP24_offsets;
    }
    int get_SP24_size()
    {
        return (*SP24Blocks).size();
    }
    int get_num_24Sparse_blocks()
    {
        return num_24Sparse_blocks;
    }
    int get_num_dense_blocks()
    {
        return num_dense_blocks;
    }
    int get_num_dense_elems()
    {
        return num_dense_elems;
    }
    int get_num_24Sparse_elems()
    {
        return num_24Sparse_elems;
    }
    int get_num_VerySparse_elems()
    {
        return num_VerySparse_elems;
    }
        int get_num_VerySparse_elems_split()
    {
        return num_VerySparse_elems_split;
    }
    int get_num_VerySparse_elems_total()
    {
        return num_VerySparse_elems_total;
    }
    int get_num_VerySparse_elems_blocks()
    {
        return num_VerySparse_blocks;
    }
};
#endif
#ifndef SPLITTER_HPP
#define SPLITTER_HPP
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_set>
#include "SparseCoo.hpp"
#include "CSR.hpp"
#include <cassert>

using namespace std;

class Splitter
{
    private:
    //// Genneral info
    CSR_Matrix<COMPUTETYPE>* A_CSR;
    int tile_size;
    int NNZThreshold;
    int total_tiles; // total number of light and heacy tiles

    //// heavy matrix
    int num_heavy_tiles;
    int num_heavy_elems;
    SparseCoo<COMPUTETYPE>* heavyMatrix;

    //// light matrix (COO)
    int num_light_tiles;
    int num_light_elems;
    SparseCoo<COMPUTETYPE>* lightMatrix;

    /// splits the matrix into heavyMatrix and lightMatrix by comparing any tile's nnz with NNZThreshold
    void split()
    {
        //1. Explore the A_CSR and generates and fills the hash tabel this.map
        //  - The key in the hash table is "row panel ID"
        //  - The value is another hash table in which 
        //     - the key is "column block ID"
        //     - the value is nnz
        //  - Each block is tile_size x tile_size

        unordered_map<int, unordered_map<int,int>> map;
        //// Go over A_CSR and fill the hash table map
        for (int row = 0; row < A_CSR->nrows; row++)
        {
            int blockRowIdx = row / tile_size;
            if(map.find(blockRowIdx) == map.end())
            {
                unordered_map<int, int> cols_map;
                map[blockRowIdx] = cols_map; // generate an empty row panel
            }
            int start = A_CSR->rowPtr[row];
            int end = A_CSR->rowPtr[row + 1];
            for (int elemIdx = start; elemIdx < end; elemIdx++)
            {
                int col = A_CSR->cols[elemIdx];
                int blockColIdx = col / tile_size;
                if(map[blockRowIdx].find(blockColIdx) == map[blockRowIdx].end()) // if block does not exist in the row-panel
                {
                    map[blockRowIdx][blockColIdx] = 1;
                    total_tiles++;
                }
                else
                {
                    map[blockRowIdx][blockColIdx] = map[blockRowIdx][blockColIdx] + 1;
                }
            }
        }
        //// 2. Go over the map and compute the num_heavy_tiles and num_light_tiles.
        for(auto &_rpanel : map)
        {
            for(auto &_blk :_rpanel.second)
            {
                if( _blk.second < NNZThreshold)
                    num_light_tiles++;
                else
                    num_heavy_tiles++;
            }
        }

        //// 3. Go over all A_CSR and use the map to split
        //// traverse each non-zero and find its block from map compare its block with nnzThreshold
        //// assign it either to a heavyMatrix  or lightMatrix
        for (int row = 0; row < A_CSR->nrows; row++) 
        {
            int start = A_CSR->rowPtr[row];
            int end = A_CSR->rowPtr[row + 1];

            for (int elemIdx = start; elemIdx < end; elemIdx++)
            {
                int col = A_CSR->cols[elemIdx];
                int blockRowIdx = row / tile_size;
                int blockColIdx = col / tile_size;

                if(map.find(blockRowIdx) != map.end())
                {
                    auto &_rowpanel = map[blockRowIdx];
                    if(_rowpanel.find(blockColIdx) != _rowpanel.end())
                    {
                        int blkNNZ = _rowpanel[blockColIdx];
                        if(blkNNZ < NNZThreshold) // if the blk is light assign the element into the lightMatrix
                        {
                            num_light_elems++;
                            lightMatrix->Insert_Next(row, col, A_CSR->values[elemIdx]);
                        }
                        else // if the blk is heavy assign the element into the heavyMatrix
                        {
                            num_heavy_elems++;
                            heavyMatrix->Insert_Next(row, col, A_CSR->values[elemIdx]);
                        }
                    }
                    else
                    {
                        assert(false && "Error: could not find the col-panel of the element, somthing went wrong while generating the map!");
                    }
                }
                else
                {
                    assert(false && "Error: could not find the row-panel of the element, somthing went wrong while generating the map!");
                }
            }
        }
    }
        
    public:
    Splitter(CSR_Matrix<COMPUTETYPE> * A_CSR, int tile_size, int NNZThreshold)
    {
        this->A_CSR = A_CSR;
        this->tile_size = tile_size;
        this->NNZThreshold = NNZThreshold;
        total_tiles = 0;
        num_heavy_tiles = 0;
        num_heavy_elems = 0;
        num_light_tiles = 0;
        num_light_elems = 0;

        lightMatrix = new SparseCoo<COMPUTETYPE>(A_CSR->nrows, A_CSR->ncols, 0);
        heavyMatrix = new SparseCoo<COMPUTETYPE>(A_CSR->nrows, A_CSR->ncols, 0);
        cout<<"\n==================================\nSplitting...\n";
        split();
    }
    ~Splitter()
    {}
    /// @brief total number of fill tiles
    /// @return 
    int get_total_tiles()
    {
        return total_tiles;
    }

    /// @brief number of heavy tiles
    int get_num_heavy_tiles()
    {
        return num_heavy_tiles;
    }

    /// @brief number of heavy elements
    int get_num_heavy_elems()
    {
        return num_heavy_elems;
    }

    /// @brief number of light tiles
    int get_num_light_tiles()
    {
        return num_light_tiles;
    }

    /// @brief number of light elements
    int get_num_light_elems()
    {
        return num_light_elems;
    }

    /// @brief the light matric in coo format
    SparseCoo<COMPUTETYPE>* get_heavyMatrix()
    {
        return heavyMatrix;
    }
    
    /// @brief the heavy matrix in coo format
    SparseCoo<COMPUTETYPE>* get_lightMatrix()
    {
        return lightMatrix;
    }
};
#endif
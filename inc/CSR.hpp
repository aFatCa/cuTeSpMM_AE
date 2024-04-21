#ifndef CSR_HPP
#define CSR_HPP
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include "SparseCoo.hpp"
#include "SMA.hpp"
#include <unordered_set>
#define Debug false
#include <cstring> // Include this header for std::memset

template<class T>
bool sortCSR(const data_point<T>& L, const data_point<T>& R)
{
    if(L.row != R.row) return L.row<R.row;
    if(L.col != R.col) return L.col<R.col;
    return true;
}

struct sortbysec {
    bool operator()(const std::pair<int,int> &left, const std::pair<int,int> &right) {
        return left.second > right.second;
    }
};
template <class T>
void print_dense(T* mat, int nrows, int ncols)
{
    cout<<"printing the dense Matrix "<<nrows<<"x"<<ncols<<"\n\n";
    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            cout<<(float) mat[ncols * i + j];
            if(j != ncols - 1)
                cout<<",";
        }
        cout<<endl;
    }
    cout<<"=================================="<<endl;
}
template <class DType>
struct CSR_Matrix
{
    int nrows;
    int ncols;
    int nnz;
    
    DType* values;
    int* cols;
    int* rowPtr;

    // /*Specific to BSCR, only valid to access after calling to_BCSR() method*/
    int R; //row blocksize
    int C; //column blocksize
    int BCSR_nrows;
    int BCSR_ncols;
    int BCSR_nblks; // number of non-empty blocks in BCSR
    DType* BCSR_values;
    int* BCSR_cols;
    int* BCSR_rowPtr;
   
    DType* values_sorted;
    int* cols_sorted;
    int* rowPtr_sorted;

    CSR_Matrix(){}
    
    CSR_Matrix(SparseCoo<DType>& S)
    {
        coo2CSR(S);
    }
    CSR_Matrix(DType* S,int m,int n)
    {
        DenseToCSR(S, m, n);
    }
    ~CSR_Matrix()
    {
        if(nnz > 0)
        {
            delete[] values;
            delete[] cols;
            delete[] rowPtr;
        }
    }
    void print_topk(int k = 10)
    {
        cout<<"\nTop"<<k<<" CSR element are:\n";
        cout<<"nrows:"<<nrows<<"  ncols:"<<ncols<<" nnz:"<<nnz<<endl;
        
        cout<<endl<<"Vals:"<<endl;
        for(int i = 0;i< min(k, nnz);i++)
        {
            cout<<(float)values[i]<<" ";
        }

        cout<<endl<<"Cols:"<<endl;
        for(int i = 0;i< min(k, nnz);i++)
        {
            cout<<cols[i]<<" ";
        }
        
        cout<<endl<<"rowPtr:"<<endl;
        for(int i = 0;i<= nrows;i++)
        {
            cout<<rowPtr[i]<<" ";
        }
        
        
        cout<<"\n==================================\n";
    }

    void print_BCSR(int k = 10)
    {
        cout<<"nrows:"<<nrows<<"  ncols:"<<ncols<<" nnz:"<<nnz<<endl;
        cout<<"BCSR_blockSize_R:"<<R<<" BCSR_blockSize_C:"<<C<<endl;
        cout<<"BCSR_nrows:"<<BCSR_nrows<<"  BCSR_ncols:"<<BCSR_ncols<<" BCSR_nblks:"<<BCSR_nblks<<endl;
        cout<<endl<<"BCSR_values:"<<endl;
        for(int i = 0;i< R*C*BCSR_nblks;i++)
        {
            cout<<BCSR_values[i]<<" ";
        }
        cout<<endl<<"BCSR_cols:"<<endl;
        for(int i = 0;i< BCSR_nblks;i++)
        {
            cout<<BCSR_cols[i]<<" ";
        }
        cout<<endl<<"BCSR_rowPtr:"<<endl;
        for(int i = 0;i<= BCSR_nrows;i++)
        {
            cout<<BCSR_rowPtr[i]<<" ";
        }
        cout<<"\n==================================\n";
    }

    void print_topk_sorted(int k = 10)
    {
        cout<<"\nTop"<<k<<" CSR element are:\n";
        cout<<"nrows:"<<nrows<<"  ncols:"<<ncols<<" nnz:"<<nnz<<endl;
        
        cout<<endl<<"Vals:"<<endl;
        for(int i = 0;i< min(k, nnz);i++)
        {
            cout<<values_sorted[i]<<" ";
        }

        cout<<endl<<"Cols:"<<endl;
        for(int i = 0;i< min(k, nnz);i++)
        {
            cout<<cols_sorted[i]<<" ";
        }
        
        cout<<endl<<"rowPtr:"<<endl;
        for(int i = 0;i<= nrows;i++)
        {
            cout<<rowPtr_sorted[i]<<" ";
        }
        
        
        cout<<"\n==================================\n";
    }
    void coo2CSR (SparseCoo<DType>& S)
    {
        // Sort COO in CSR way
        // We need to update SparseCoo
        sort(S.data_vector.begin(),S.data_vector.end(),sortCSR<DType>);
        // S.Print_topk(20);
        // Allocate space
        nrows = S.dimY;
        ncols = S.dimX;
        nnz = S.nnz;
        rowPtr = new int [nrows + 1];
        memset (rowPtr, 0, sizeof (int) * (nrows + 1));
        cols = new int [nnz];
        values = new DType[nnz];

        for (int i = 0; i < nnz; i++)
        {
            values[i] = S.data_vector[i].val;
            cols[i]   = S.data_vector[i].col;
            rowPtr[S.data_vector[i].row + 1]++;
        }
        for (int i = 0; i < nrows; i++)
        {
            rowPtr[i+1] += rowPtr[i];
        }
    }
    void sort_rows()
    {
        cout<<"==================================\n";
        cout<<"Sorting the  matrix rows based on NNZ...\n";
        vector<pair<int,int>> index_nnz;
        for(int i=0; i< nrows; i++)
        {
            index_nnz.push_back({i,rowPtr[i+1]-rowPtr[i]});
        }
        sort(index_nnz.begin(), index_nnz.end(), sortbysec());
        cout<<"sort finished\n";
        vector<int> sorted_vals;
        vector<int> sorted_cols;
        vector<int> sorted_rptr;
        sorted_rptr.push_back(0);
        for(int i = 0; i < nrows; i++)
        {
            int start = rowPtr[index_nnz[i].first];
            for(int index = start; index< start+index_nnz[i].second; index++)
            {
                sorted_vals.push_back(values[index]);
                sorted_cols.push_back(cols[index]);
            }
            sorted_rptr.push_back(sorted_rptr.back()+index_nnz[i].second);
        }
        values_sorted = new DType[nnz];
        cols_sorted = new int[nnz];
        rowPtr_sorted = new int[nrows+1];
        std::copy(sorted_vals.begin(), sorted_vals.end(), values_sorted);
        std::copy(sorted_cols.begin(), sorted_cols.end(), cols_sorted);
        std::copy(sorted_rptr.begin(), sorted_rptr.end(), rowPtr_sorted);
        cout<<"==================================\n";
    }
    int cut(int Min_NNZ_to_use_TCU,int nnz_moving_avg_window_size, float nnz_moving_avg_threashold, int cut_margin)
    {
        /*
         * cut is decided based on the density.
         * Density in the cut is proportional to the row nnz, so we consider the row nnz to calculate the cut
         * 1. the cut is considered to be the global max fall that is an indicator for the place that density reduced the most
         * 2. if a cut not found the algorithm considers 0.95*nrows as the cut to bias toward using TCU.
         * 3. if a cut found but very early then it will be replaced with the cut margin
         * TODO: there might be a case that even the nnz is very low that it does not worth tcu... (like first row nnz in sorted array < 16)
         */

        cout<<"Cut finding ...\n";
        //1. find the global cut
        SMA nnz_moving_average(nnz_moving_avg_window_size);
        nnz_moving_average.add(rowPtr_sorted[1]);

        int global_cut = -1;
        float global_fall_ratio = 1000000;
        for (int row = 1 ; row < nrows-1; row++)
        {
            int next_row_nnz = (rowPtr_sorted[row+2] - rowPtr_sorted[row+1]); 
            if(Debug)
            {
                cout<<"NxtRowNNZ:"<<next_row_nnz<<" mAVG:"<<nnz_moving_average.avg()<<endl;
                cout<<"NxtRowNNZ/mAVG:"<<next_row_nnz/nnz_moving_average.avg()<<" mAVGThresh:"<<nnz_moving_avg_threashold<<endl;
                cout<<"===========================\n";
                getchar();
            }
            int cut_candidate = row;
            float candidate_fall_ratio = next_row_nnz/nnz_moving_average.avg();
            if ( cut_candidate > nnz_moving_avg_window_size && candidate_fall_ratio < nnz_moving_avg_threashold)
            {
                if(Debug)
                    cout<<"cut_candidate:"<<cut_candidate<<" candidate_fall_ratio:"<<candidate_fall_ratio<<" global_fall_ratio:"<<global_fall_ratio<<endl;
                if(candidate_fall_ratio <= global_fall_ratio)
                {
                    global_fall_ratio = candidate_fall_ratio;
                    global_cut = cut_candidate;
                    if(Debug)
                        cout<<"global_cut:"<<global_cut<<endl;
                }
            }
            nnz_moving_average.add(next_row_nnz);
        }

        //2. if did not find a cut then bias the cut toward using TCU: e.g cut = 0.95 * nrows
        if(global_cut)
        {
            cout<<"A cut was not found with the current nnz_moving_avg_threashold:"<<nnz_moving_avg_threashold<<endl;
            cout<<"Decrease the nnz_moving_avg_threashold to increase the chance of finding candidate cuts."<<endl;
            cout<<"Cut is biased toward using TCU cut = 0.95*"<<nrows<<endl;
            global_cut = 0.95 * nrows; 
        }

        //3. cut margin
        if (global_cut < cut_margin * nrows)
        {
            cout<<"Global cut:"<<global_cut<<" is before the cut_margin*nrows:"<<cut_margin * nrows<<endl;
            cout<<"Global cut replaced with cut_margin*nrows"<<endl;
            global_cut = cut_margin * nrows;
        }
        return global_cut;
    }
    int cut_basic(float basic_cut_density_threshold)
    {
        int current_row_nnz = rowPtr_sorted[1];
        int dense_tile_nnz = current_row_nnz;
        int TCU_NNZ = basic_cut_density_threshold * nnz;
        int row = 1;
        for (;row < nrows-1; row++)
        {
            current_row_nnz = (rowPtr_sorted[row+1] - rowPtr_sorted[row]);
            if(dense_tile_nnz + current_row_nnz <= TCU_NNZ)
                dense_tile_nnz += current_row_nnz;
            else
                break;
        }
        return row + 1; 
    }
    
    void loadSortedToDense(COMPUTETYPE * tile, int K, int R, int C)
    {
        //Loads the first K rows of the CSR into a R by C tile 
        for(int row = 0 ; row < K && row < nrows-1; row++ )
        {
            int start = rowPtr_sorted[row];
            int end = rowPtr_sorted[row+1];
            for(int elem_index = start; elem_index< end; elem_index++)
            {
                tile[row*C + cols_sorted[elem_index]] = (COMPUTETYPE) (values_sorted[elem_index]);
            }
        }
    }

    template <class T>
    void ToDense(T* dense)
    {
        //Loads the CSR to Dense 
        for(int row = 0 ; row < nrows; row++ )
        {
            int start = rowPtr[row];
            int end = rowPtr[row+1];
            for(int elem_index = start; elem_index< end; elem_index++)
            {
                dense[row*ncols + cols[elem_index]] = (T) (values[elem_index]);
            }
        }
	    cout<<"CSR to Dense complete\n";
    }

    void loadSortedToCSR(CSR_Matrix * tile, int R)
    {
        // //Loads from the row index R to end into another CSR tile
        vector<int>tileRptr;
        vector<int>tileCols;
        vector<DType>tileValues;
        for(int row = R; row < nrows;row++)
        {
            int v = rowPtr_sorted[row]-rowPtr_sorted[R];
            tileRptr.push_back(v);
        }
        tileRptr.push_back(rowPtr_sorted[nrows]-rowPtr_sorted[R]);

        for(int elemIndex = rowPtr_sorted[R]; elemIndex< rowPtr_sorted[nrows];elemIndex++)
        {
            tileCols.push_back(cols_sorted[elemIndex]);
            tileValues.push_back(values_sorted[elemIndex]);
        }
        tile->nrows = tileRptr.size();
        tile->ncols = ncols;
        tile->nnz = tileValues.size();
        tile->values = new DType[tile->nnz];
        tile->cols = new int[tile->nnz];
        tile->rowPtr = new int[tile->nrows];
        std::copy(tileRptr.begin(), tileRptr.end(), tile->rowPtr);
        std::copy(tileValues.begin(), tileValues.end(), tile->values);
        std::copy(tileCols.begin(), tileCols.end(), tile->cols);
    }

    int calulate_blocked_ELL_columns(int ELL_blocksize)
    {
        // ELL-Blocked columns = blocksize * max_row_block_number
        // max_row_block_number is maximum number of blocks in all row blocks
        
        // 1. calculate the blocks in each rowblock
        int numBlocksY = ((nrows + ELL_blocksize - 1)/ELL_blocksize);// ceil(nrows/ELL_blocksize)
        vector<unordered_set<int>>colIdx (numBlocksY, unordered_set<int>()); // colIdx[row] is a set containing the nonempty block offsets
        
        for(int row = 0; row < nrows; row++)
        {
            int RowStart = rowPtr[row];
            int RowEnd = rowPtr[row+1];
            for(int elementIdx = RowStart; elementIdx < RowEnd; elementIdx++)
            {
                int col = cols[elementIdx];
                int blockRowIdx = row / ELL_blocksize;
                int blockColIdx = col / ELL_blocksize;
                
                if(colIdx[blockRowIdx].find(blockColIdx) == colIdx[blockRowIdx].end()) // if non-empty blockOffset not already added to the relevant set
                {
                    colIdx[blockRowIdx].insert(blockColIdx);
                }
            }
        }

        //2. return blocksize * max_row_block_number
        int max_row_block_number = 0;
        for(auto s : colIdx)
        {
            max_row_block_number = (s.size() > max_row_block_number)? s.size():max_row_block_number;
        }
        return max_row_block_number * ELL_blocksize;
    }

    /*  converts CSR to ELL-Blocked format
     *  @params IN  ELL_blocksize: size of each Ell block
     *  @params IN  blocked_ELL_columns: number of columns in the ELL_values 
     *  @params OUT ELL_columnsIdx: offset of nonempty blocks in the Ell representation
     *  @params IN  numBlocksX: blocked_Ell_columns / Ell_blocksize 
     *  @params IN  numBlocksY: number of rows of the original matrix / Ell_blocksize
     *  @params OUT ELL_values: nonempty values in the Ell representation.
     *  look into the example figure under https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-spmat-create-blockedell
     */
    template <class T>
    void csrToEllBlocked(int ELL_blocksize, int blocked_ELL_columns, int* ELL_columnsIdx, int numBlocksX, int numBlocksY, T* ELL_values)
    {
        // 1. compute the blocked_ELL column indices 
        vector<unordered_set<int>>colIdx (numBlocksY, unordered_set<int>()); // colIdx[row] is a set containing the "nonempty block" offsets
        // cout<<"numBlocksY:"<<numBlocksY<<" numBlocksX:"<<numBlocksX<<" ELL_blocksize:"<<ELL_blocksize<<" blocked_ELL_columns:"<<blocked_ELL_columns<<endl;
        int total_number_of_nonempty_blocks = 0;
        for(int row = 0; row < nrows; row++)
        {
            int RowStart = rowPtr[row];
            int RowEnd = rowPtr[row+1];
            for(int elementIdx = RowStart; elementIdx < RowEnd; elementIdx++)
            {
                int col = cols[elementIdx];
                int blockRowIdx = row / ELL_blocksize;
                int blockColIdx = col / ELL_blocksize;
                
                if(colIdx[blockRowIdx].find(blockColIdx) == colIdx[blockRowIdx].end()) // if non-empty blockOffset not already added to the relevant set
                {
                    colIdx[blockRowIdx].insert(blockColIdx);
                    total_number_of_nonempty_blocks++;
                }
            }
        }
        assert(total_number_of_nonempty_blocks <= (numBlocksX*numBlocksY));
        
        // copy the sets into ELL_columnsIdx
        int i = 0;
        for(auto s : colIdx)
        {
            vector<int> v(s.begin(), s.end());
            sort(v.begin(), v.end());
            for(int elem : v)
            {
                ELL_columnsIdx[i] = elem;
                i++;
            }
        }

        //2. compute the blocked_ELL values array
        /*
         * Observation: the ellValues is similar to CSR.values but
         * not identical. e.g. if all values of the first block in 
         * the first block row is 0 except the last one 
         * i.e. 14, ell-values keeps all the 0s in the block 
         * but csr.values is not.
         */
        // CSR to dense
        T* A_Dense = new T[nrows * ncols];
        memset(A_Dense, 0, sizeof(T) * (nrows * ncols)); 
        ToDense(A_Dense);
        // cout<<"dense format::\n";
        // print_dense(A_Dense, nrows, ncols);
        // compute ELL_values using ELL_columnsIdx and A_Dense
        for(int blockRow = 0; blockRow < numBlocksY; blockRow++) // traverse the ELL_columnsIdx
        {
            for(int blockCol = 0; blockCol < numBlocksX; blockCol++)
            {
                int NZ_blockCol = ELL_columnsIdx[blockRow * numBlocksX + blockCol];
                if(NZ_blockCol >= 0) // if a nonempty block
                {
                    for(int row = blockRow * ELL_blocksize; row < (blockRow + 1) * ELL_blocksize; row++)
                    {
                        for(int col =  NZ_blockCol * ELL_blocksize; col <  (NZ_blockCol + 1) * ELL_blocksize; col++)
                        {
                            int empty_blocks_to_skip = (NZ_blockCol - blockCol);
                            int ELL_values_col = col - empty_blocks_to_skip * ELL_blocksize;
                            ELL_values[row * blocked_ELL_columns + ELL_values_col] = A_Dense[row * ncols + col];
                            // cout<<row<<","<<ELL_values_col<<","<<(float)A_Dense[row * ncols + col]<<"\n";
                        }
                    }
                }
                else // if a padding block 
                {
                    break;
                }
            }
        }
        cout<<"csrToEllBlocked completed.\n";
    }

    void DenseToCSR(DType* S, int m, int n)
    {
        nrows = m;
        ncols = n;
        vector<DType> vls;
        vector<int> columns;
        vector<int> rptr;
        bool firstElementInRowFound = false;
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                float t = (float)S[i*n+j];
                // cout<<"t:"<<t<<endl;
                if(t != 0)
                {
                    if(!firstElementInRowFound)
                    {
                        firstElementInRowFound = true;
                        rptr.push_back(vls.size());
                    }
                    nnz++;
                    vls.push_back(S[i*n+j]);
                    columns.push_back(j);
                }
            }
            firstElementInRowFound = false;
        }
        rptr.push_back(vls.size());
        values = new DType[nnz];
        cols = new int[nnz];
        rowPtr = new int[nrows+1];
        std::copy(vls.begin(), vls.end(), values);        
        std::copy(columns.begin(), columns.end(), cols);
        std::copy(rptr.begin(), rptr.end(), rowPtr);
    }
    
    /* swaps row r1 and r2*/
    void swapRows(int r1, int r2)
    {
        // handle values and cols
        
        /*
        0.....r1s...r1e.......r2s..........r2e.....nnz-1
        0.....r2s..........r2e.......r1s...r1e.....nnz-1
        */
        int* cls    = new int[nnz];
        DType* vls = new DType[nnz];
        int next = 0;
        for(int i = 0; i < rowPtr[r1]; i++)
        {
            vls[next] = values[i];
            cls[next] = cols[i];
            next++;
        }
        // cout<<"next:"<<next<<endl;
        for(int i = rowPtr[r2]; i < rowPtr[r2+1]; i++)
        {
            vls[next] = values[i];
            cls[next] = cols[i];
            next++;
        }
        // cout<<"next:"<<next<<endl;

        for(int i = rowPtr[r1+1]; i < rowPtr[r2]; i++) 
        {
            vls[next] = values[i];
            cls[next] = cols[i];
            next++;
        }
        // cout<<"next:"<<next<<endl;

        for(int i = rowPtr[r1]; i < rowPtr[r1+1]; i++) 
        {
            vls[next] = values[i];
            cls[next] = cols[i];
            next++;
        }
        // cout<<"next:"<<next<<endl;

        for(int i = rowPtr[r2+1]; i < rowPtr[nrows]; i++) 
        {
            vls[next] = values[i];
            cls[next] = cols[i];
            next++;
        }
        
        if(next != nnz)
        {
            cout<<"Error: nnz should not change after row reordering!\n";
            cout<<"r1:"<<r1<<" r2:"<<r2<<endl;
            cout<<"r1s:"<<rowPtr[r1]<<" r1e:"<<rowPtr[r1+1]<<endl;
            cout<<"r2s:"<<rowPtr[r2]<<" r2e:"<<rowPtr[r2+1]<<endl;
            cout<<"next:"<<next<<" nnz:"<<nnz<<endl;
            cout<<"len r1:"<<rowPtr[r1+1]-rowPtr[r1]<<" len r2:"<<rowPtr[r2+1]-rowPtr[r2]<<endl;
            cout<<"===\n";
            assert(false);
        }
        for(int i = 0; i<nnz; i++)
        {
            values[i] = vls[i];
            cols[i] = cls[i];
        }
        // =======
        // handle rowPtr

        // rowPtr[0:r1] won't change
        // rowPtr[r1+1:r2] increses by diff = (rowPtr[r2]-rowPtr[r1])
        int diff = ((rowPtr[r2+1]-rowPtr[r2])-(rowPtr[r1+1]-rowPtr[r1]));
        for(int i = r1+1; i < r2+1; i++)
        {
            rowPtr[i] += diff;
        }
        // rowPtr[r2+1:] won't change
        
        // =======
        // cleanup

        delete[] cls;
        delete[] vls;
    }

    /* Returns the position similarity of two rows r1 and r2
     * i.e. similarity = number of elelments in that share same colum
     * in the CSR we basically find the commonality of col values of the 2 rows
     * O(len(r1) + len(r2)) ~= o(K)
     */
    int row_simlarity(int r1, int r2)
    {
        int sim = 0;
        int row1Start = rowPtr[r1];
        int row1End = rowPtr[r1+1];
        int row2Start = rowPtr[r2];
        int row2End = rowPtr[r2+1];
        if((row1End - row1Start) < (row2End - row2Start)) // if row1 has less non-zero
        {
            unordered_set<int> searchSet;
            // add all the elements of the smaller row to the set
            for(int i = row1Start; i < row1End; i++)
            {
                searchSet.insert(cols[i]);
            }

            // go through all the elements of the larger row and increment the sim by 1 for each match
            for(int i = row2Start; i < row2End; i++)
            {
                if(searchSet.find(cols[i]) != searchSet.end()) 
                { 
                    // cout<<"i:"<<i<<" col:"<<cols[i]<<" is common\n";
                    sim++;
                }
            }
        }
        else
        {
            // cout<<"r2\n";
            unordered_set<int> searchSet;
            for(int i = row2Start; i < row2End; i++)
            {
                searchSet.insert(cols[i]);
            }
            // go through all the elements of the larger row and increment the sim by 1 for each match
            for(int i = row1Start; i < row1End; i++)
            {
                if(searchSet.find(cols[i]) != searchSet.end())
                { 
                    // cout<<"i:"<<i<<" col:"<<cols[i]<<" is common\n";
                    sim++;
                }
            }
        }
        if(sim > min((row1End - row1Start),(row2End - row2Start)))
        {
            cout<<"Error: similarity="<<sim<<" it should never exceed the min row length="<<min((row1End - row1Start),(row2End - row2Start))<<"!!\n";
            assert(false);
        }
        return sim;
    }

    /* Searches for the most similar row to row r in A[Row+1:nrows]. O(m*k) */
    int find_most_similar(int Row)
    {
        int max_sim = 0;
        int max_sim_index = Row;
        for(int r = Row+1; r<nrows; r++) // O((nrows-Row)*k) = o(m*k)
        {
            int sim = row_simlarity(Row,r); // O(k)
            if( sim > max_sim)
            {
                max_sim = sim;
                max_sim_index = r;
            }
        }
        return max_sim_index;
    }

    /* Gets a 4x4 block and returns the sum of the nnz each row needs to get perfect */
    int Cost(int br, int bj)
    {
        int cost = 0;
        for(int row = br; row<br+4; row++)
        {
            int nnz = 0;
            //or(int col=bj; col<bj+4; col++) if(block[i][j]) nnz++; // counts the nnz in row i
            cost += (abs(nnz-2)); // nnz=3 => cost +1 is this ok?
        }
        return cost;
    }

    /* Computes the cost of a KxK block */
    int BlockCost(int BR, int BC,int K)
    {
        int cost = 0;
        for(int br=BR; br<BR+K; br+=4)
        { 
            for(int bc = BC; bc<BC+K; bc+=4) 
            {
                cost+= Cost(br, bc);
            }
        }
        return cost; 
    }

    /*  */
    void swap_cols_in_row_panel(int panel, int b1, int b2)
    {

    }

    /*
    * Compute the number of occupied RxC blocks in a matrix
    *
    * Input Arguments:
    *   int  R             - row blocksize
    *   int  C             - column blocksize
    *
    * Output Arguments:
    *   int  num_blocks    - number of blocks
    *
    * Note:
    *   Complexity: Linear
    *
    */
    int csr_count_blocks(const int R, const int C)
    {
        vector<int> mask(ncols/C + 1,-1);
        int n_blks = 0;
        for(int i = 0; i < nrows; i++){
            int bi = i/R;
            for(int jj = rowPtr[i]; jj < rowPtr[i+1]; jj++){
                int bj = cols[jj]/C;
                if(mask[bj] != bi){
                    mask[bj] = bi;
                    n_blks++;
                }
            }
        }
        return n_blks;
    }

    /*
    * Convert a CSR matrix to BCSR format
    *
    * Input Arguments:
    *   int  R               - row blocksize
    *   int  C               - column blocksize
    *
    * Output Arguments:
    *   DType* BCSR_values[nrows/R + 1] - block row pointer
    *   int* BCSR_cols[nnz]             - column indices
    *   int* BCSR_rowPtr[nnz]           - nonzero blocks
    *
    * Note:
    *   Complexity: Linear
    */
    void to_BCSR(int R_, int C_)
    {
        assert( nrows % R_ == 0 );
        assert( ncols % C_ == 0 );
        R = R_;
        C = C_;
        std::vector<DType*> blocks(ncols/C + 1, (DType*)0 );


        BCSR_nrows = nrows / R;
        BCSR_ncols = ncols / C;
        BCSR_nblks = csr_count_blocks(R,C);

        int BlockSize = R*C;
        int n_blks = 0;

        BCSR_rowPtr = new int [BCSR_nrows + 1];
        memset (BCSR_rowPtr, 0, sizeof (int) * (BCSR_nrows + 1));

        BCSR_cols = new int [BCSR_nblks];
        memset (BCSR_cols, 0, sizeof (int) * (BCSR_nblks));
        
        BCSR_values = new DType[(uint32_t)BCSR_nblks * R * C];
        memset (BCSR_values, 0, sizeof (DType) * ((uint32_t)BCSR_nblks * R * C));

        for(int bi = 0; bi < BCSR_nrows; bi++)
        {
            for(int r = 0; r < R; r++)
            {
                int i = R*bi + r;  //row index
                for(int jj = rowPtr[i]; jj < rowPtr[i+1]; jj++)
                {
                    int j = cols[jj]; //column index
                    int bj = j / C;
                    int c  = j % C;
                    if( blocks[bj] == 0 )
                    {
                        blocks[bj] = BCSR_values + BlockSize*n_blks;
                        BCSR_cols[n_blks] = bj;
                        n_blks++;
                    }
                    *(blocks[bj] + C*r + c) += values[jj];
                }
            }
            for(int jj = rowPtr[R*bi]; jj < rowPtr[R*(bi+1)]; jj++)
            {
                blocks[cols[jj] / C] = 0;
            }
            BCSR_rowPtr[bi+1] = n_blks;
        }
    }
};

#endif

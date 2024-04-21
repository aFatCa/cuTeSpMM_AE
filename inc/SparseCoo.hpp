#ifndef SPARSECOO_HPP
#define SPARSECOO_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cassert>
#include <algorithm>
#include <cublas_v2.h>

using namespace std;

template<class T>
struct data_point{
    int row;
    int col;
    T val;
    data_point(int x, int y, T v)
    {
        row=x;
        col=y;
        val=v;
    }
    
    void print()
    {
        cout<< row<<","<<col<<","<<val<<endl;
    }
};
template<class T>
bool sortOrder(const data_point<T>& L, const data_point<T>& R)
{
    if(L.row != R.row) return L.row<R.row;
    if(L.col != R.col) return L.col<R.col;
    return true;
}
template <class DType>
struct SparseCoo
{
    //Coo containers
    vector<data_point<DType>> data_vector;
    int dimY; // nrows
    int dimX; // ncols
    int nnz;
    int* rows;
    int* cols;
    DType* values;
    unordered_map<int, vector<int>> tiles;

    /*Specific to BCOO, only valid to access after calling to_BCOO() method*/
    int R; //row blocksize
    int C; //column blocksize
    int BCOO_nrows; // nrows/R
    int BCOO_ncols; // ncols/C
    int BCOO_nblks; // number of non-empty blocks in BCOO
    
    int* BCOO_rows;
    int* BCOO_cols;
    DType* BCOO_values; // BCOO_nblks * (R * C)

    ~SparseCoo()
    {
        if(rows != NULL)
            delete[] rows;
        if(cols != NULL)
            delete[] cols;
        if(values != NULL)
            delete[] values;

        if(BCOO_rows != NULL)
            delete[] BCOO_rows;
        if(BCOO_cols != NULL)
            delete[] BCOO_cols;
        if(BCOO_values != NULL)
            delete[] BCOO_values;
    }
    SparseCoo()
    {
        rows = NULL;
        cols = NULL;
        values = NULL;
        BCOO_rows = NULL;
        BCOO_cols = NULL;
        BCOO_values = NULL;
    }

    SparseCoo(string FilePath)
    {
        rows = NULL;
        cols = NULL;
        values = NULL;
        BCOO_rows = NULL;
        BCOO_cols = NULL;
        BCOO_values = NULL;
        ifstream infile;
        try{
            infile.open(FilePath);
        }
        catch(exception e)
        {
            cout<<e.what();
        }
        cout<<"\n==================================\nReading the Sparse matrix...\n";

        string line;
        // is matrix symetric?
        getline(infile, line);
        bool symmetric = (line.find("symmetric") != std::string::npos )? true:false;
        if(symmetric)
            cout<<"!!>>>>> Note: Matrix is symmetric <<<<<<!!"<<endl;
        cout<<"Skipped line: "<<line<<endl;

        // skip comments     
        while (getline(infile, line))
        {
            if(line[0] == '%')
                cout<<"Skipped line: "<<line<<endl; 
            else
                break;
        }
        
        // line should contain dimY,dimX,NNZ at this point
        istringstream iss(line);
        iss >> dimY >> dimX >> nnz;
        //read the matrix data
        while (getline(infile, line))
        {
            istringstream iss(line);
            unsigned int x, y;
            double vt=1.0;
            iss >> x >> y;
            if (iss)
            {
                iss >> vt;
            }
            DType v = (DType) vt; // type cast in to Dtype
            data_point<DType> n(x-1,y-1,v);// -1 to start from 0 instead of 1
            data_vector.push_back(n);

            if(symmetric && x != y)
            {
                data_point<DType> n(y-1,x-1,v);// -1 to start from 0 instead of 1
                data_vector.push_back(n);
                nnz++;
            }
        }
        infile.close();
        sort(data_vector.begin(),data_vector.end(),sortOrder<DType>);
        cout<<"The Sparse matrix was read successfully.\n";
        cout<<"dimX:"<<dimX<<" dimY:"<<dimY<<" NNZ:"<<nnz<<endl;
        cout<<"==================================\n";

    }
    SparseCoo(int dY, int dX, int nnz)
    {
        dimY = dY;
        dimX = dX;
        this->nnz = nnz; 
        rows = NULL;
        cols = NULL;
        values = NULL;
        BCOO_rows = NULL;
        BCOO_cols = NULL;
        BCOO_values = NULL;
    }
    void Print_topk(int k=10)
    {
        cout<<"\nTop "<<k<<" COO element are:\n";
        cout<<"dimY:"<<dimY<<"  dimX:"<<dimX<<" nnz:"<<nnz<<endl;
        
        cout<<"Rows:"<<endl;
        for(int i = 0;i< min(k, nnz);i++)
        {
            cout<<data_vector[i].row<<" ";
        }
        
        cout<<endl<<"Cols:"<<endl;
        for(int i = 0;i< min(k, nnz);i++)
        {
            cout<<data_vector[i].col<<" ";
        }
        
        cout<<endl<<"Vals:"<<endl;
        for(int i = 0;i< min(k, nnz);i++)
        {
            cout<<data_vector[i].val<<" ";
        }
        cout<<"\n==================================\n";
    }
    void tile_2D(int TI, int TK)
    {
        for(int i = 0; i< data_vector.size();i++)
        {
            int tile_row = data_vector[i].row/TI;
            int tile_col = data_vector[i].col/TK;
            int offset = tile_row*TK+tile_col;
            if (tiles.find(offset) != tiles.end()) 
            {
                tiles[offset].push_back(i);
            }
            else 
            {
                vector<int> t;
                t.push_back(i);
                tiles[offset] = t;
            }
        }
    }
    template<class T>
    void zero_fill(T* adj, uint32_t m, uint32_t n)
    {
        for(int elem_Idx = 0; elem_Idx < nnz; elem_Idx++)
        {
            data_point<DType> elem = data_vector[elem_Idx];
            uint64_t offset = (uint64_t)elem.row * dimX + elem.col;
            if(elem.row>=m || elem.col>=n)
            {
                cout<<" m:"<<m<<" n:"<<n<<endl;
                cout<<"elem.row: "<<elem.row<<" elem.col: "<<elem.col<<endl;
                assert(false && "out of bound access!");
            }
            adj[offset] = (T) elem.val;
        }
    }
    int getX()
    {
        return dimX;
    }
    int getY()
    {
        return dimY;
    }
    //// these methods are addded to be able to grow the matrix not from a file but from data points.
    void setDim(int dimY,int dimX)
    {
        this->dimY = dimY;
        this->dimX = dimX;
        nnz = 0;
    }
    void Insert_Next(int row,int col, DType val)
    {
        data_point<DType> n(row,col,val);
        data_vector.push_back(n);
        nnz ++;
    }
    void prepare_for_cusparse()
    {
        rows = new int[nnz];
        cols = new int[nnz];
        values = new DType[nnz];
        for(int i = 0; i<data_vector.size(); i++)
        {
            rows[i] = data_vector[i].row;
            cols[i] = data_vector[i].col;
            values[i] = (DType)data_vector[i].val;
        }
    }
    /// save the coo in .mtx format
    void save_mtx(string out_filePath)
    {
        cout<<"Saving filename:"<<out_filePath<<endl;
        ofstream outfile;
        try
        {
            outfile.open(out_filePath);
        }
        catch(exception e)
        {
            cout<<e.what();
        }
        if(nnz>0)
        {
            outfile<<"%%MatrixMarket matrix coordinate real general"<<endl; //// mtx metadata
            outfile<<dimY<<" "<<dimX<<" "<<nnz<<endl; //// header
            for(int i = 0; i < data_vector.size(); i++)
            {
                outfile<<1 + data_vector[i].row<<" "<<1 + data_vector[i].col<<" "<<(float)data_vector[i].val<<endl; //// header
            }
        }
        outfile.close();
    }

    void toBCOO(int R_, int C_)
    {
        cout<<"Converting COO to BCOO...\n";
        assert( dimY % R_ == 0 );
        assert( dimX % C_ == 0 );
        R = R_;
        C = C_;
        
        BCOO_nrows  = dimY / R;
        BCOO_ncols  = dimX / C;

        BCOO_nblks  = nnz / (R*C); 
        vector<DType> BCOORows;
        vector<DType> BCOOCols;
        vector<DType> BCOOVals;

        prepare_for_cusparse();
        unordered_map<int, DType> dok; // used for computing the BCOO_values
        for(int i = 0; i<nnz; i++)
        {
            int r = rows[i];
            int c = cols[i];
            int offset  = r*dimX + c;
            dok[offset] = values[i];
        }
        /// generating BCOO.rows, BCOO.cols, and BCOO_values
        unordered_set<int> blkOffsets; // keeps the fillBlks offset
        for(int i = 0; i<nnz; i++)
        {
            int r = rows[i];
            int c = cols[i];
            int BR = r/R;
            int BC = c/C;
            int BO = BR * BCOO_ncols + BC; // BlockOffset of the element(r,c)
            if(blkOffsets.find(BO)==blkOffsets.end()) // if BO not already visited
            {
                blkOffsets.insert(BO);
                BCOORows.push_back(BR);
                BCOOCols.push_back(BC);
                //// inserting the block value into the BCOO_values
                for(int rr = BR*R; rr<(BR+1)*R; rr++)
                {
                    for(int cc = BC*C; cc<(BC+1)*C; cc++)
                    {
                        int offset  = rr *  dimX + cc;
                        if(dok.find(offset) != dok.end())
                        {
                            BCOOVals.push_back(dok[offset]);
                        }
                        else
                        {
                            cout<<"key not found:"<<offset<<endl;
                            assert(false);
                        }
                    }
                }
            }
        }
        BCOO_rows = new int[BCOORows.size()];
        for(int i = 0; i<BCOORows.size(); i++)
        {
            BCOO_rows[i] = BCOORows[i];
        }

        BCOO_cols = new int[BCOOCols.size()];
        for(int i = 0; i<BCOOCols.size(); i++)
        {
            BCOO_cols[i] = BCOOCols[i];
        }

        BCOO_values = new DType[BCOOVals.size()];
        for(int i = 0; i<BCOOVals.size(); i++)
        {
            BCOO_values[i] = BCOOVals[i];
        }
        cout<<"convertion completed.\n";
    }

    /// @brief printing the first k blks of BCOO
    /// @param k 
    void print_BCOO_k(int k=1)
    {
        cout<<"=========\nprinting BCOO info:\nR:"<<R<<" C:"<<C<<endl;
        cout<<"BCOO_nrows:"<<BCOO_nrows<<" BCOO_ncols:"<<BCOO_ncols<<" BCOO_nblks:"<<BCOO_nblks<<endl;
        cout<<"\nrows:\n";
        for(int i = 0; i<min(k, BCOO_nblks); i++)
        {
            cout<<BCOO_rows[i]<<",";
        }
        cout<<"\ncols:\n";
        for(int i = 0; i<min(k, BCOO_nblks); i++)
        {
            cout<<BCOO_cols[i]<<",";
        }
        cout<<"\nvals:\n";
        for(int i = 0; i<min(nnz, k*R*C); i++)
        {
            cout<<(float)BCOO_values[i]<<",";
        }
        cout<<endl;
    }
};

#endif

#include "inc.h"
class MTX_READER{
public:
    unsigned int M;
    unsigned int K;
    unsigned int nnz;
    unsigned int *rowPtr;
    unsigned int *cols;
    float * vals;
    MTX_READER(std::string file_path){
        nnz = 0;
        std::string line;
        std::ifstream infile;
        infile.open(file_path);
        getline(infile, line);
        bool symmetric = (line.find("symmetric") != std::string::npos );
        while(getline(infile, line)){
            if(line[0] == '%')
                continue;
            else
                break;
        }
        std::string word;
        std::istringstream ss(line);
        ss >> word;
        M = std::stoi(word);
        ss >> word;
        K = std::stoi(word);
        ss >> word;
        M = ((M - 1)/TM + 1) * TM;
        K = ((K - 1)/TK + 1) * TK;
        rowPtr = new unsigned int [M + 1];
        memset(rowPtr, 0, (M + 1) * sizeof(unsigned int));
        while(getline(infile,line)){
            std::istringstream ss(line);
            ss >> word;
            unsigned int s = std::stoi(word);
            ss >> word;
            unsigned int e = std::stoi(word);
            ss >> word;
            rowPtr[s]++;
            if (symmetric && (s != e)){
                rowPtr[e]++;
            }
        }
        infile.close();
        for(unsigned int i=0; i<M; ++i){
            rowPtr[i + 1] += rowPtr[i];
        }
        cols = new unsigned int [rowPtr[M]];
        vals = new float[rowPtr[M]];
        infile.open(file_path);
        while(getline(infile, line)){
            if(line[0] == '%')
                continue;
            else
                break;
        }
        unsigned int *counters = new unsigned int [M];
        memset(counters, 0, M * sizeof(unsigned int));
        while(getline(infile,line)){
            std::istringstream ss(line);
            ss >> word;
            unsigned int s = std::stoi(word);
            ss >> word;
            unsigned int e = std::stoi(word);
            ss >> word;
            float v = 1.0f;
            s = s - 1;
            e = e - 1;
            unsigned int pos = rowPtr[s];
            unsigned int index = counters[s];
            cols[pos + index] = e;
            vals[pos + index] = v;
            counters[s]++;
            nnz++;
            if(symmetric && (s != e)){
                pos = rowPtr[e];
                index = counters[e];
                cols[pos + index] = s;
                vals[pos + index] = 1.0f;
                counters[e]++;
                nnz++;
            }
        }
        delete [] counters;
    }
};
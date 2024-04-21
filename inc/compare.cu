//
// Created by lizhi on 2/10/23.
//
#include <mma.h>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <stdio.h>
inline void chkerr(cudaError_t code, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) <<" at line "<<line<<std::endl;
        exit(-1);
    }
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
void read_file(unsigned int *Ys, unsigned int *Xs, float *Vs, std::string input_file){
    unsigned int line_index = 0;
    std::string line;
    std::ifstream infile;
    infile.open(input_file);
    while(getline(infile,line)){
        std::istringstream ss(line);
        std::string word;
        ss >> word;
        unsigned int s = std::stoi(word);
        ss >> word;
        unsigned int e = std::stoi(word);
        ss >> word;
        float v = std::stof(word);
        Ys[line_index] = s;
        Xs[line_index] = e;
        Vs[line_index] = v;
        line_index++;
    }
    infile.close();
}
struct cmpPos {
    bool operator()(const Pos & a, const Pos & b) const {
        if(a.r < b.r){
            return true;
        }
        if(a.r == b.r){
            if(a.c < b.c){
                return true;
            }else{
                return false;
            }
        }
        return false;
    }
};


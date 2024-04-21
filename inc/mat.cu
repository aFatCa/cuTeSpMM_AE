#include "compare.cu"
#define TB 128
#define TM 16
#define TN 64
#define TK 8
#define NUM_TM_BRICKS ((TB - 1)/TM + 1)
#define NUM_TK_BRICKS ((TB - 1)/TK + 1)
class Brick{
public:
    float *vals;
    Brick(){
        vals = new float[TM*TK];
        memset(vals, 0, sizeof(float) * TM * TK);
    }
    /*~Brick(){
        delete []vals;
    }*/
};

class BLOCK{
public:
    unsigned int M = NUM_TM_BRICKS;
    unsigned int N = NUM_TK_BRICKS;
    unsigned int numActiveBricks;
    unsigned int *brickPos;
    float * vals;
    std::map<Pos, Brick, cmpPos> bricks;
    BLOCK(){
        vals = new float[TB*TB];
        brickPos = new unsigned int [NUM_TM_BRICKS*NUM_TK_BRICKS];
        numActiveBricks = 0;
    }
    /*~BLOCK(){
        delete [] vals;
        delete [] brickPos;
    }*/
    void toArray(){
        numActiveBricks = bricks.size();
        unsigned int idx = 0;
        for(std::map<Pos, Brick>::iterator it=bricks.begin(); it!=bricks.end(); ++it){
            Pos p = it->first;
            brickPos[idx]  =  p.r * N + p.c;
            memcpy(&vals[idx * TM * TK], it->second.vals, TM*TK*sizeof(float));
            idx++;
        }
        //bricks.clear();
    }
};
class Mat{
public:
    unsigned int M;
    unsigned int N;
    unsigned int *rowPtr;
    unsigned int *cols;
    float *vals;
    unsigned int *activeBricksCount;
    unsigned int *activeBricks;

    unsigned int *rowPtrDev;
    unsigned int *colsDev;
    float *valsDev;
    unsigned int *activeBricksCountDev;
    unsigned int *activeBricksDev;
    unsigned int numActiveBlocks;
    std::map<Pos, BLOCK, cmpPos> blocks;
    void map_coo(unsigned int *Ys, unsigned int *Xs, float *Vs, unsigned int NNZs){
        for(unsigned int i=0;i<NNZs;++i){
            unsigned int r1 = Ys[i] / TB;
            unsigned int c1 = Xs[i] / TB;
            unsigned int r2 = (Ys[i] - r1 * TB) / TM;
            unsigned int c2 = (Xs[i] - c1 * TB) / TK;
            unsigned int r3 = (Ys[i] - r1 * TB) % TM;
            unsigned int c3 = (Xs[i] - c1 * TB) % TK;
            float v = Vs[i];
            Pos pos1(r1, c1);
            Pos pos2(r2, c2);
            if(blocks.find(pos1) == blocks.end()){
                BLOCK block;
                Brick brick;
                brick.vals[r3 * TK + c3] = v;
                block.bricks.insert(std::pair<Pos, Brick> (pos2, brick));
                blocks.insert(std::pair<Pos, BLOCK> (pos1, block));
                numActiveBlocks++;
                rowPtr[r1+1]++;
            }else{
                if(blocks[pos1].bricks.find(pos2) == blocks[pos1].bricks.end()){
                    Brick brick;
                    brick.vals[r3 * TK + c3] = v;
                    blocks[pos1].bricks.insert(std::pair<Pos, Brick> (pos2, brick));
                }else{
                    blocks[pos1].bricks[pos2].vals[r3 * TK + c3] = v;
                }
            }
        }
        for(unsigned int i=0;i<M;++i){
            rowPtr[i+1]+=rowPtr[i];
        }
        //std::cout<<"after coo map"<<std::endl;
    }
    void check_correctness(unsigned int NNZs, unsigned int Y, unsigned int X, unsigned int *Ys, unsigned int *Xs, float *Vs){
        float *dense = new float[Y*X];
        memset(dense, 0, sizeof(float)*Y*X);
        for(unsigned int i=0;i<M;++i){
            for(unsigned int j=rowPtr[i];j<rowPtr[i+1];j++){
                unsigned int col = cols[j];
                unsigned int K = activeBricksCount[j];
                for(unsigned int k=0;k<K;k++){
                    unsigned int brick_pos = activeBricks[j*NUM_TK_BRICKS*NUM_TM_BRICKS+k];
                    unsigned int r = brick_pos/NUM_TK_BRICKS;
                    unsigned int c = brick_pos%NUM_TK_BRICKS;
                    float *brick_vals = &vals[j*TB*TB+k*TM*TK];
                    r = i * TB + r * TM;
                    c = col * TB + c * TK;
                    for(unsigned int ii=0;ii<TM;++ii){
                        for(unsigned int kk=0;kk<TK;++kk){
                            unsigned int r_original = r + ii;
                            unsigned int c_original = c + kk;
                            dense[r_original * X + c_original] = brick_vals[ii*TK+kk];
                        }
                    }
                }
            }
        }
        float diff1 = 0.0f;
        float total2 = 0.0f;
        float total1 = 0.0f;
        for(unsigned int i=0;i<NNZs;++i){
            unsigned int r = Ys[i];
            unsigned int c = Xs[i];
            float v = Vs[i];
            total1+= v;
            //std::cout<<dense[r*X+c]<<","<<v<<std::endl;
            diff1 += abs(dense[r*X+c] - v);
        }
        for(unsigned int i=0;i<Y;++i){
            for(unsigned int j=0;j<X;++j){
                total2 += dense[i*X+j];
            }
        }
	delete [] dense;
        std::cout<<diff1<<","<<abs(total2 - total1)<<std::endl;
    }
    Mat(unsigned int *Ys, unsigned int *Xs, float *Vs, unsigned int Y, unsigned int X, unsigned int NNZs){
    // Mat(std::string input_file, unsigned int Y, unsigned int X, unsigned int NNZs){
        numActiveBlocks = 0;
        M = (Y - 1)/TB + 1;
        N = (X - 1)/TB + 1;
        rowPtr = new unsigned int [M + 1];
        memset(rowPtr, 0, sizeof(unsigned int)*(M+1));
        // unsigned int *Ys= new unsigned int [NNZs];
        // unsigned int *Xs = new unsigned int [NNZs];
        // float *Vs = new float [NNZs];
        // read_file(Ys, Xs, Vs, input_file);
        map_coo(Ys, Xs, Vs, NNZs);
        vals = new float[numActiveBlocks*TB*TB];
        activeBricksCount = new unsigned int [numActiveBlocks];
        cols = new unsigned int [numActiveBlocks];
        activeBricks = new unsigned int [NUM_TM_BRICKS*NUM_TK_BRICKS*numActiveBlocks];
        unsigned int idx = 0;
        for(std::map<Pos, BLOCK>::iterator i=blocks.begin(); i!=blocks.end(); ++i){
            Pos pos = i->first;
            i->second.toArray();
            memcpy(&vals[idx*TB*TB], i->second.vals, TB*TB*sizeof(float));
            cols[idx] = pos.c;
            activeBricksCount[idx] = i->second.numActiveBricks;
            memcpy(&activeBricks[idx*(NUM_TM_BRICKS*NUM_TK_BRICKS)], i->second.brickPos, (NUM_TM_BRICKS*NUM_TK_BRICKS)*sizeof(unsigned int));
	    i->second.bricks.clear();
            idx++;
        }
        //std::cout<<"here after map coo"<<std::endl;
        //blocks.clear();
        //std::cout<<"here after clear"<<std::endl;

        //check_correctness(NNZs, Y, X, Ys, Xs, Vs);
        delete []Xs;
        delete []Ys;
        delete []Vs;
	blocks.clear();
        chkerr(cudaMalloc(&rowPtrDev, (M+1)*sizeof(unsigned int)),__LINE__);
        chkerr(cudaMalloc(&valsDev, (numActiveBlocks*TB*TB)*sizeof(float)),__LINE__);
        chkerr(cudaMalloc(&activeBricksCountDev, (numActiveBlocks)*sizeof(unsigned int)),__LINE__);
        chkerr(cudaMalloc(&colsDev, (numActiveBlocks)*sizeof(unsigned int)),__LINE__);
        chkerr(cudaMalloc(&activeBricksDev, (numActiveBlocks*(NUM_TM_BRICKS*NUM_TK_BRICKS))*sizeof(unsigned int)),__LINE__);

        chkerr(cudaMemcpy(rowPtrDev, rowPtr, (M+1)*sizeof(unsigned int), cudaMemcpyHostToDevice), __LINE__);
        chkerr(cudaMemcpy(valsDev, vals, (numActiveBlocks*TB*TB)*sizeof(float), cudaMemcpyHostToDevice), __LINE__);
        chkerr(cudaMemcpy(activeBricksCountDev, activeBricksCount, (numActiveBlocks)*sizeof(unsigned int),
                          cudaMemcpyHostToDevice), __LINE__);
        chkerr(cudaMemcpy(colsDev, cols, (numActiveBlocks)*sizeof(unsigned int), cudaMemcpyHostToDevice), __LINE__);
        chkerr(cudaMemcpy(activeBricksDev, activeBricks, (numActiveBlocks*(NUM_TM_BRICKS*NUM_TK_BRICKS))*sizeof(unsigned int), cudaMemcpyHostToDevice), __LINE__);
	delete [] rowPtr;
	delete [] vals;
	delete [] activeBricksCount;
	delete [] cols;
	delete [] activeBricks;
    }
};
void gemm(float *a, float *b, float *c, unsigned int M, unsigned int K, unsigned int N){
#pragma omp parallel for
    for(unsigned int i=0;i<M;++i){
        for(unsigned int j=0;j<N;++j){
            float result = 0.0f;
            for(unsigned int k=0;k<K;++k){
                result += a[i*K+k] * b[k*N+j];
            }
            c[i*N+j] = result;
        }
    }
}
void read_graph(std::string file, float *arr, unsigned int M, unsigned int K){
    memset(arr, 0, M*K*sizeof(float));
    std::string line;
    std::ifstream infile;
    infile.open(file);
    while(getline(infile,line)){
        std::istringstream ss(line);
        std::string word;
        //std::cout<<line<<std::endl;
        ss >> word;
        unsigned int s = std::stoi(word);
        ss >> word;
        unsigned int e = std::stoi(word);
        ss >> word;
        float v = std::stof(word);
        arr[s*K+e] = v;
    }
    infile.close();
}
/*int main(int argc, char *argv[]){
    srand((unsigned int)time(NULL));
    unsigned int M = atoi(argv[1]);
    unsigned int K = atoi(argv[2]);
    unsigned int nnz = atoi(argv[3]);
    std::string input_file = std::string(argv[4]);
    Mat mat(input_file,M,K,nnz);
    return 0;
}*/



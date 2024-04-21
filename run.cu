#include "spmm.cu"
#define MAX_N 512
#ifndef ERROR_CHECK
#define ERROR_CHECK 1
#endif
#include <nvToolsExt.h>

int main(int argc, char *argv[]){
    srand((unsigned int)time(NULL));
    std::string mtx_file = std::string(argv[1]);
    MTX_READER * mtxReader = new MTX_READER(mtx_file);
    unsigned int M = mtxReader->M;
    unsigned int K = mtxReader->K;
    unsigned int nnz = mtxReader->nnz;
    std::string base_filename = mtx_file.substr(mtx_file.find_last_of("/\\") + 1);
    unsigned int feature_size_list[3] = {32, 128, 512};
    double start = omp_get_wtime();
    HRPB hrpb(M, K, mtxReader->rowPtr, mtxReader->cols, mtxReader->vals);
    double end = omp_get_wtime();
    double preprocessing_time = (end - start)*1000;
    float * c_host = new float[M * MAX_N];
    float * CSR_result_host = new float[M * MAX_N];
    float * b = new float[K * MAX_N];
    #pragma omp parallel for
    for(unsigned int i=0;i<K*MAX_N;++i){
        b[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0f;
    }
    for(int n=0; n<3; ++n){
        unsigned int N = feature_size_list[n];
        //////////////////// NVTX MARKERS start
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        // int tm = TM, tn=TN;
        eventAttrib.message.ascii = ("cute_"+std::to_string(TM)+"_"+std::to_string(TN)+"_"+std::to_string(N)).c_str();
        nvtxRangePushEx(&eventAttrib);
        ////////////////////

        float * b_device;
        chkerr(cudaMalloc(&b_device, K*N*sizeof(float)), __LINE__);
        chkerr(cudaMemcpy(b_device, b, K*N*sizeof(float), cudaMemcpyHostToDevice), __LINE__);
        float * c_device;
        chkerr(cudaMalloc(&c_device, M*N*sizeof(float)), __LINE__);
        chkerr(cudaMemset(c_device, 0, M*N*sizeof(float)), __LINE__);

        SPMM_TC spmmTc(M, K, N, hrpb.blockedRowPtr);
        unsigned int * data = hrpb.blocksDevice;
        unsigned long long int *sizePtr = hrpb.sizePtrDevice;
        unsigned int *active_cols_vec = hrpb.activeColsDevice;
        cudaMemset(c_device, 0, M*N*sizeof(float));
        spmmTc.run(data, b_device, c_device, active_cols_vec, sizePtr);
        //////////////////// NVTX MARKERS start
        nvtxRangePop();
        chkerr(cudaMemcpy(c_host, c_device, M*N*sizeof(float), cudaMemcpyDeviceToHost), __LINE__);
        chkerr(cudaFree(c_device), __LINE__);
        chkerr(cudaFree(b_device), __LINE__);
        chkerr(cudaDeviceSynchronize(), __LINE__);
        #if ERROR_CHECK
        memset (CSR_result_host, 0, sizeof (float) * ((uint64_t) M*MAX_N));
        GPU_CSR_cuSPARSE_NEW(mtxReader->rowPtr, mtxReader->cols, mtxReader->vals,
                             mtxReader->nnz, b, CSR_result_host, M, K, N, 1);
        chkerr(cudaDeviceSynchronize(), __LINE__);
        float diff1 = check_gemm_correctness_coo(CSR_result_host, c_host, M, N);
        //std::cout<<"CORRECTNESS CHECK: "<<diff1<<std::endl;
        #endif
        //std::cout<<"-,filepath,filename,TM,TN,m,k,n,nnz, prep_time(ms)\n";
        //std::cout<<"dummy,"<<mtx_file<<","<<base_filename<<","<<TM<<","<<TN<<","<<M<<","<<K<<","<<N<<","<<nnz<<","<<preprocessing_time<<std::endl;
    }
    delete []c_host;
    delete []CSR_result_host;
    delete []b;
    return 0;
}


import codecs
import os, sys
import pandas as pd
from pathlib import Path
suite_sparse_root = './ae_matrices'
if not os.path.exists(suite_sparse_root):
    print('suite sparse root error')
    exit(0)
matrices = os.listdir(suite_sparse_root)
reader = codecs.open('meta.data', 'r', 'utf-8')
lines = reader.readlines()
shape_info = {}
for line in lines:
    parts = line.split(',')
    matrix = parts[0]
    nnz = int(parts[-1])
    shape_info[matrix] = nnz
gflops_list = []
n_list = [32, 128, 512]
for matrix in matrices:
    nnz = shape_info[matrix]
    df = pd.read_csv(f"{matrix}.csv")
    #filtered_df = df[(df["Kernel Name"].str.contains("csrmm_alg2_kernel")) | (df["Kernel Name"].str.contains("SpMM"))]
    metric_value_list = df["Metric Value"].tolist()
    metric_value_list_clean = []
    for value in metric_value_list:
        clean_value = value.replace(',','')
        metric_value_list_clean.append(int(clean_value))
    for index, n in enumerate(n_list):
        num_flops = nnz * n * 2
        cute_time = metric_value_list_clean[index * 7]
        csr_time = sum(metric_value_list_clean[index * 7 + 1:(index + 1) * 7])/2
        cute_gflops = num_flops/cute_time
        csr_gflops = num_flops/csr_time
        gflops_list.append((matrix, n, cute_gflops, csr_gflops))
print('matrix, N(feature size), cuTeSpMM GFLOPS, cuSparse CSR GFLOPS\n')
for gflops_info in gflops_list:
    matrix = gflops_info[0]
    n = gflops_info[1]
    cute_gflops = round(gflops_info[2], 2)
    csr_gflops = round(gflops_info[3], 2)
    print(f'{matrix}, {n}, {cute_gflops}, {csr_gflops}')


    


import numpy as np
import sys
import os
matrices = os.listdir("ae_matrices")
tcgnn_ae_folder = "ae_matrices_tcgnn_format"
for matrix in matrices:
    filefolder = os.path.join("ae_matrices", matrix)
    filename = os.path.join(filefolder, f'{matrix}.mtx')
    src = []
    dst = []
    nrows = 0
    ncols = 0
    nnz = 0
    with open(filename, 'r') as fl:
        output = []
        mtxFirstLine = True
        for line in fl:
            if len(line) == 0 or line.startswith("%") or line.startswith("#"):
                continue ## skip empty lines and comment lines
            if  mtxFirstLine:
                tuple = line.split()
                nrows = int(tuple[0])
                ncols = int(tuple[1])
                nnz = int(tuple[2])
                mtxFirstLine = False
                continue ## skip the header in the .mtx files
            tuple = line.split()
            src.append(int(tuple[0])-1)
            dst.append(int(tuple[1])-1)
    num_nodes =np.asarray(max(nrows, ncols))
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    save_name = os.path.join(tcgnn_ae_folder, f'{matrix}.npz')
    np.savez_compressed(save_name, src_li=src, dst_li=dst, num_nodes= num_nodes)

import numpy as np
import sys

filename=sys.argv[1]
print(filename)
if filename.endswith(".npz"):
    data = np.load(sys.argv[1])
    src = data['src_li']
    dst = data['dst_li']
    nnz = len(src)
    output = []
    output.append("%%MatrixMarket matrix coordinate real general")
    max_m = 0
    max_n = 0
    for i in range(len(src)):
        row = src[i] + 1
        col = dst[i] + 1
        if(row > max_m):
            max_m = row
        if(col > max_n):
            max_n = col
        output.append(str(row) +" "+str(col))
    info = str(max_m)+ " " + str(max_n) + " "+str(int(nnz))
    output.insert(1, info)
    print(output[0])
    print(output[1])
    # for line in output[2:]:
    #     print (line + " 1.0")
    output_filename = filename.split(".")[0]+".mtx"
    with open(output_filename, 'w') as f:
        for line in output:
            f.write(line + '\n')

elif filename.endswith(".mtx"):
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
    np.savez_compressed(filename.split(".")[0], src_li=src, dst_li=dst, num_nodes= num_nodes)

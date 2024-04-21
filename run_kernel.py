import codecs
import os, sys
from pathlib import Path
suite_sparse_root = './ae_matrices'
if not os.path.exists(suite_sparse_root):
    print('suite sparse root error')
    exit(0)
matrices = os.listdir(suite_sparse_root)
done_matrices = []
for index, matrix in enumerate(matrices):
    print(f'{index} out of {len(matrices)} done')
    if matrix in done_matrices:
        continue
    matrix_path = os.path.join(suite_sparse_root, matrix)
    matrix_path = os.path.join(matrix_path, f'{matrix}.mtx')
    command = f"ncu --metrics gpu__time_duration.sum --csv ./run_cute_16_4 {matrix_path}"
    with open('tmp.csv', 'w') as file:
        os.system(f"{command} > tmp.csv")
    reader = codecs.open('tmp.csv', 'r', 'utf-8')
    result_lines = reader.readlines()
    reader.close()
    with open(f'{matrix}.csv', 'w') as out:
        for result_line in result_lines:
            if '.mtx' not in result_line and 'PROF' not in result_line:
                out.write(result_line)
    os.remove('tmp.csv')
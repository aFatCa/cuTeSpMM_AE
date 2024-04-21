./1_run_bSpMM.py| tee 1_run_bSpMM.log
./1_log2csv.py 1_run_bSpMM.log
python 2_combine_results.py
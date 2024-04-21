#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import numpy as np 
import subprocess

hidden = 16
block_density = 0.0025
block_scale = 1 / block_density
tile_size = 512

dataset = [
        ('citeseer'	                , 3327	    ,  9464   ),  
        ('cora' 	        	, 2708	    , 10858   ),  
        ('pubmed'	        	, 19717	    , 88676   ),      
        ('ppi'	            		, 56944	    , 818716 ),   
        
        ('PROTEINS_full'             , 43471            , 162088) ,   
        ('OVCAR-8H'                  , 1890931          , 3946402) , 
        ('Yeast'                     , 1714644          , 3636546) ,
        ('DD'                        , 334925           , 1686092) ,
        ('YeastH'                    , 3139988          , 6487230) ,   
        
        ( 'amazon0505'          , 410236  , 4878875),
        ( 'artist'              , 50515	  , 1638396),
        ( 'com-amazon'          , 548551  , 1851744),
        ( 'soc-BlogCatalog'	, 88784	  , 2093195),      
        ( 'amazon0601'  	, 403394  , 3387388), 
]

for data, node, edges in dataset:
    print("dataset={}".format(data))
    ntimes = max(block_scale * edges / (tile_size * tile_size), 1) 
    result = subprocess.run(["python", "2_cusparse_test.py"], stdout=subprocess.PIPE)
    output = result.stdout.decode()
    res = float(output.split()[9]) * ntimes
    print("{:.3f}".format(res))
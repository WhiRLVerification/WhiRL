from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from tensorflow.python.saved_model import tag_constants


from marabou_K_query_sd import *




import sys
import os

def main():
    if len(sys.argv) not in [4]:
        print("usage:", sys.argv[0], "<pb_filename_prefix> [k] [download_time]")
        exit(0)

    pb_filename_format = '{}_{}.{}'
    #.format(checkpoints_dir,model_name,idx,suffix) # checkpoints_dir+"/"+model_name++str(0)+
    idx = 0
    pb_filename_prefix = sys.argv[1]
    k = int(sys.argv[2])
    download_time = float(sys.argv[3])
    # print(pb_filename_format.format(pb_filename_prefix,idx, "pb"))
    results = []
    while os.path.isfile(pb_filename_format.format(pb_filename_prefix,idx, "pb")):
        pb_filename = pb_filename_format.format(pb_filename_prefix,idx, "pb")
        print("-------------------------------------------------")
        print("                               "+pb_filename)
        print("                    checkpoint_"+str(idx))
        print("-------------------------------------------------")
        print(pb_filename)
        results.append(k_test(pb_filename, k, download_time))
        # basic_test(pb_filename, len(sys.argv) == 3)
        idx+=1
    print(results)

if __name__ == "__main__":
    main()

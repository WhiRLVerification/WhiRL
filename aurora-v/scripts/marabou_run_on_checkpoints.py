from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from tensorflow.python.saved_model import tag_constants


from marabou_pcc_tf import *




import sys
import os

def main():
    if len(sys.argv) not in [2, 3]:
        print("usage:", sys.argv[0], "<pb_filename_prefix> [-l]")
        exit(0)

    pb_filename_format = '{}_{}.{}'
    #.format(checkpoints_dir,model_name,idx,suffix) # checkpoints_dir+"/"+model_name++str(0)+
    idx = 0
    pb_filename_prefix = sys.argv[1]
    # print(pb_filename_format.format(pb_filename_prefix,idx, "pb"))
    while os.path.isfile(pb_filename_format.format(pb_filename_prefix,idx, "pb")):
        if idx == 2:
            idx+=1
        pb_filename = pb_filename_format.format(pb_filename_prefix,idx, "pb")
        print("-------------------------------------------------")
        print("                    checkpoint_"+str(idx))
        print("-------------------------------------------------")
        print(pb_filename)
        print("=========================-marabou_test-=========================")
        marabou_test(pb_filename, len(sys.argv) == 3)
        print("=========================-marabou_test_2-=========================")
        marabou_test_2(pb_filename, len(sys.argv) == 3)
        idx+=1

if __name__ == "__main__":
    main()

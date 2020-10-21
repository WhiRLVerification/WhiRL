import numpy as np
import sys
M = A_DIM = 6
S_INFO = 6
S_LEN = 8
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]
OUTPUT_TO_FILE = True
import os
# # split_0 = tflearn.fully_connected(inputs[:, 0:1, -1]    , 128, activation='relu')
# # split_1 = tflearn.fully_connected(inputs[:, 1:2, -1]    , 128, activation='relu')
# # split_2 = tflearn.conv_1d(inputs[:, 2:3, :]     , 128, 4, activation='relu')
# # split_3 = tflearn.conv_1d(inputs[:, 3:4, :]     , 128, 4, activation='relu')
# # split_4 = tflearn.conv_1d(inputs[:, 4:5, :A_DIM], 128, 4, activation='relu')
# # split_5 = tflearn.fully_connected(inputs[:, 4:5, -1]    , 128, activation='relu')
#
# # state = [np.zeros((S_INFO, S_LEN))]
# # state = np.roll(state, -1, axis=1)[0]
# # state[0, -1] = 1 # VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality                 # Last chunk bit rate -      [1]  l~t
# # state[1, -1] = 2 # buffer_size / BUFFER_NORM_FACTOR  # 10 sec #                                             # Current buffer size -      [1]  b~t
# # state[2, -1] = 3 # float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / s                         # Past chunk throughput -    [k]  x~t
# # state[3, -1] = 4 #float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec                                      # Past chunk download time - [k]  τ~t
# # state[4, :A_DIM] = 5 # np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte                      # Next chunk sizes -         [m]  n~t
# # state[5, -1] = 6 #np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)  # Number of chunks left      [1]  c~t
#
# # last_chunk_bit_rate, current_buffer_size, past_chunk_throughput, past_chunk_download_time, next_chunk_sizes, number_of_chunks_left

def prep_input_for_query (networkInputVars,k):
    last_chunk_bit_rate_arr = []
    current_buffer_size_arr = []
    past_chunk_throughput_arr = []
    past_chunk_download_time_arr = []
    next_chunk_sizes_arr = []
    number_of_chunks_left_arr = []
    all_inputs = set()
    used_inputs  = set()
    #print("networkInputVars", networkInputVars)
    assert (len(networkInputVars) == k)
    for inputVars in networkInputVars:
     #   print ("inputVrs shape = ", inputVars.shape)
     #   print("inputVars:")
     #   print(inputVars)
        all_inputs = all_inputs.union(inputVars.flatten())
        last_chunk_bit_rate = inputVars[:, 0:1, -1] [0]      #[1]  l~t
        assert (len(last_chunk_bit_rate) == 1)
        last_chunk_bit_rate_arr.append(last_chunk_bit_rate)
        for i in last_chunk_bit_rate:
            used_inputs.add(i)
       #     print("last_chunk_bit_rate")
        #    print(i)

        current_buffer_size = inputVars[:, 1:2, -1] [0]        #[1]  b~t
        assert (len(current_buffer_size) == 1)
        current_buffer_size_arr.append(current_buffer_size)
        for i in current_buffer_size:
            used_inputs.add(i)
         #   print(i)

        past_chunk_throughput = inputVars[:, 2:3, :] [0][0]    #[k]  x~t
        assert (len(past_chunk_throughput) == S_LEN)
        past_chunk_throughput_arr.append(past_chunk_throughput)
        for i in past_chunk_throughput:
            used_inputs.add(i)
          #  print(i)

        past_chunk_download_time = inputVars[:, 3:4, :] [0][0]    #[k]  τ~t
        assert (len(past_chunk_download_time) == S_LEN)
        past_chunk_download_time_arr.append(past_chunk_download_time)
        for i in past_chunk_download_time:
            used_inputs.add(i)
           # print(i)


        next_chunk_sizes = inputVars[:, 4:5, :A_DIM] [0][0]       #[m]  n~t
        assert (len(next_chunk_sizes) == M)
        next_chunk_sizes_arr.append(next_chunk_sizes)
        for i in next_chunk_sizes:
            used_inputs.add(i)
            #print(i)

        number_of_chunks_left = inputVars[:, 4:5, -1] [0]       #[1]  c~t
        assert (len(number_of_chunks_left) == 1)
        number_of_chunks_left_arr.append(number_of_chunks_left)
        for i in number_of_chunks_left:
            used_inputs.add(i)
            #print(i)
    unused_inputs = all_inputs - used_inputs

    return all_inputs, used_inputs, unused_inputs, last_chunk_bit_rate_arr, current_buffer_size_arr, past_chunk_throughput_arr, \
           past_chunk_download_time_arr, next_chunk_sizes_arr, number_of_chunks_left_arr

def prep_outputs_for_query(networkOutputVars,k):
    assert (len(networkOutputVars) == k * A_DIM)
    all_outputs = np.asarray(networkOutputVars).reshape(k,A_DIM).tolist()
    return all_outputs

def handle_results(query_name,k, download_time, vals, last_chunk_bit_rate, current_buffer_size, past_chunk_throughput,past_chunk_download_time,next_chunk_sizes,number_of_chunks_left,all_outputs):
    result = 'SAT' if len(list(vals.items())) != 0 else 'UNSAT'
    pri

    nt('marabou solve run result: {} '.format(result))

    if result == 'SAT':
        for j in range(k):
            print(j, "/", k)
            print("last_chunk_bit_rate:")
            for var in last_chunk_bit_rate[j]:
                print("var", var, " = ", vals[var])

            print("current_buffer_size:")
            for var in current_buffer_size[j]:
                print("var", var, " = ", vals[var])

            print("past_chunk_throughput:")
            for var in past_chunk_throughput[j]:
                print("var", var, " = ", vals[var])

            print("past_chunk_download_time:")
            for var in past_chunk_download_time[j]:
                print("var", var, " = ", vals[var])

            print("next_chunk_sizes:")
            for var in next_chunk_sizes[j]:
                print("var", var, " = ", vals[var])

            print("number_of_chunks_left:")
            for var in number_of_chunks_left[j]:
                print("var", var, " = ", vals[var])


    if OUTPUT_TO_FILE:
        dir = query_name+"_res"
        if not os.path.isdir(dir):
            os.mkdir(dir)
        filename = dir+"/pensieve_"+query_name+"_download_time_"+str(download_time*10)+"_k_"+str(k)+".res"
        with open(filename,'w') as out_file:
            print('marabou solve run result: {} '.format(result),file=out_file)
            if result == 'SAT':
                print("Counter example: ",file=out_file)
                for j in range(k):
                    print(j, "/", k,file=out_file)
                    print("last_chunk_bit_rate:",file=out_file)
                    for var in last_chunk_bit_rate[j]:
                        print("var", var, " = ", vals[var],file=out_file)

                    print("current_buffer_size:",file=out_file)
                    for var in current_buffer_size[j]:
                        print("var", var, " = ", vals[var],file=out_file)

                    print("past_chunk_throughput:",file=out_file)
                    for var in past_chunk_throughput[j]:
                        print("var", var, " = ", vals[var],file=out_file)

                    print("past_chunk_download_time:",file=out_file)
                    for var in past_chunk_download_time[j]:
                        print("var", var, " = ", vals[var],file=out_file)

                    print("next_chunk_sizes:",file=out_file)
                    for var in next_chunk_sizes[j]:
                        print("var", var, " = ", vals[var],file=out_file)

                    print("number_of_chunks_left:",file=out_file)
                    for var in number_of_chunks_left[j]:
                        print("var", var, " = ", vals[var],file=out_file)

                    print("k =",j,"output:",file=out_file)
                    i=0
                    i_max = 0
                    max = 0
                    for var in all_outputs[j]:
                        if vals[var]>max:
                            max = vals[var]
                            i_max = i
                        i+=1
                    i=0
                    for var in all_outputs[j]:
                        if i == i_max:
                            print("var", var, " = ", vals[var], "("+str(i)+")  <== ", file=out_file)
                        else:
                            print("var", var, " = ", vals[var], "(" + str(i) + ")", file=out_file)
                        i+=1
    return result

import getopt
import parameters
import os
import environment
import numpy as np
import theano
import time
import sys
import cPickle
import writeNNet
import pg_network
import nnet
import ntpath

def script_usage():
    print('--exp_type <type of experiment> \n'
          '--num_res <number of resources> \n'
          '--num_nw <number of visible new work> \n'
          '--simu_len <simulation length> \n'
          '--num_ex <number of examples> \n'
          '--num_seq_per_batch <rough number of samples in one batch update> \n'
          '--eps_max_len <episode maximum length (terminated at the end)> \n'
          '--num_epochs <number of epoch to do the training>\n'
          '--time_horizon <time step into future, screen height> \n'
          '--res_slot <total number of resource slots, screen width> \n'
          '--max_job_len <maximum new job length> \n'
          '--max_job_size <maximum new job resource request> \n'
          '--new_job_rate <new job arrival rate> \n'
          '--dist <discount factor> \n'
          '--lr_rate <learning rate> \n'
          '--ba_size <batch size> \n'
          '--pg_re <parameter file for pg network> \n'
          '--v_re <parameter file for v network> \n'
          '--q_re <parameter file for q network> \n'
          '--out_freq <network output frequency> \n'
          '--ofile <output file name> \n'
          '--log <log file name> \n'
          '--render <plot dynamics> \n'
          '--unseen <generate unseen example> \n')


def main():

    pa = parameters.Parameters()

    type_exp = 'pg_re'  # 'pg_su' 'pg_su_compact' 'v_su', 'pg_v_re', 'pg_re', q_re', 'test'

    pg_resume = None
    v_resume = None
    q_resume = None
    log = None

    render = False
    plot = False

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hi:o:", ["exp_type=",
                      "num_res=",
                      "num_nw=",
                      "simu_len=",
                      "num_ex=",
                      "num_seq_per_batch=",
                      "eps_max_len=",
                      "num_epochs=",
                      "time_horizon=",
                      "res_slot=",
                      "max_job_len=",
                      "max_job_size=",
                      "new_job_rate=",
                      "dist=",
                      "lr_rate=",
                      "ba_size=",
                      "pg_re=",
                      "v_re=",
                      "q_re=",
                      "out_freq=",
                      "ofile=",
                      "log=",
                      "render=",
                      "unseen=",
                      "plot="])

    except getopt.GetoptError:
        script_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            script_usage()
            sys.exit()
        elif opt in ("-e", "--exp_type"):
            type_exp = arg
        elif opt in ("-n", "--num_res"):
            pa.num_resources = int(arg)
        elif opt in ("-w", "--num_nw"):
            pa.num_nw = int(arg)
        elif opt in ("-s", "--simu_len"):
            pa.simu_len = int(arg)
        elif opt in ("-n", "--num_ex"):
            pa.num_ex = int(arg)
        elif opt in ("-sp", "--num_seq_per_batch"):
            pa.num_seq_per_batch = int(arg)
        elif opt in ("-el", "--eps_max_len"):
            pa.episode_max_length = int(arg)
        elif opt in ("-ne", "--num_epochs"):
            pa.num_epochs = int(arg)
        elif opt in ("-t", "--time_horizon"):
            pa.time_horizon = int(arg)
        elif opt in ("-rs", "--res_slot"):
            pa.res_slot = int(arg)
        elif opt in ("-ml", "--max_job_len"):
            pa.max_job_len = int(arg)
        elif opt in ("-ms", "--max_job_size"):
            pa.max_job_size = int(arg)
        elif opt in ("-nr", "--new_job_rate"):
            pa.new_job_rate = float(arg)
        elif opt in ("-d", "--dist"):
            pa.discount = float(arg)
        elif opt in ("-l", "--lr_rate"):
            pa.lr_rate = float(arg)
        elif opt in ("-b", "--ba_size"):
            pa.batch_size = int(arg)
        elif opt in ("-p", "--pg_re"):
            pg_resume = arg
        elif opt in ("-v", "--v_re"):
            v_resume = arg
        elif opt in ("-q", "--q_re"):
            q_resume = arg
        elif opt in ("-f", "--out_freq"):
            pa.output_freq = int(arg)
        elif opt in ("-o", "--ofile"):
            pa.output_filename = arg
        elif opt in ("-lg", "--log"):
            log = arg
        elif opt in ("-r", "--render"):
            render = (arg == 'True')
        elif opt in ("-pl", "--plot"):
            plot = (arg == 'True')
        elif opt in ("-u", "--unseen"):
            pa.generate_unseen = (arg == 'True')
        else:
            script_usage()
            sys.exit()

    if log is not None:
        orig_stdout = sys.stdout
        f = open(log, 'w')
        sys.stdout = f
    if pg_resume is None:
        print("PG resume is empty!")
        sys.exit(1)

    pa.compute_dependent_parameters()
    repre = 'image'
    end = 'all_done'
    env = environment.Env(pa, render=render, repre=repre, end=end)
    pg_learner = pg_network.PGLearner(pa)
    net_handle = open(pg_resume, 'rb')
    net_params = cPickle.load(net_handle)
    pg_learner.set_net_params(net_params)
    outputFileName = pa.output_filename + '_' + \
        ntpath.basename(pg_resume) + '_test.pkl'
    pg_learner.write_net_to_nnet(outputFileName)
    nnetFilename = outputFileName + '.nnet'
    r = nnet.NNet(nnetFilename)
    smallHigherBound = 1.0e-10
    smallLowerBound = -1.0e-10
    for wIdx, w in enumerate(r.weights):
        smallRows = np.all((w <= smallHigherBound) &
                           (w >= smallLowerBound), axis=1)
        smallRowsIndices = np.where(smallRows == True)
        for row in smallRowsIndices[0]:
            rowBias = r.biases[wIdx][row]
            #add it's bias to all biasses of next layer since we assume fully connected
            if((wIdx + 1) < len(r.biases)):
                r.biases[wIdx + 1] = r.biases[wIdx + 1] + rowBias
        #Now we delete the lines with 'dead' neurons
        r.weights[wIdx] = np.delete(w, smallRowsIndices[0], axis=0)
        if((wIdx + 1) < len(r.weights)):
                r.weights[wIdx + 1] = np.delete(r.weights[wIdx + 1],
                                                smallRowsIndices[0], axis=1)
        r.biases[wIdx] = np.delete(r.biases[wIdx], smallRowsIndices[0], axis=0)

    # now we export the file once again after the 'dead' neuron filtration
    for wIdx, w in enumerate(r.weights):
        r.weights[wIdx] = r.weights[wIdx].transpose()
    writeNNet.writeNNet(r.weights, r.biases, r.mins, r.maxes,r.means, r.ranges, outputFileName + '_cleaned_' + '.nnet')
    if log is not None:
        sys.stdout = orig_stdout
        f.close()


if __name__ == '__main__':
    main()

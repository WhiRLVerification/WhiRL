from environment import *
from parameters import *
from maraboupy import Marabou, MarabouCore, MarabouUtils
import sys
from io import open
import os

def create_network(filename):

    input_op_names = ["input"]
    output_op_name = "y_out"

    network = Marabou.read_tf(filename, inputName=input_op_names,outputName=output_op_name)
    return network, input_op_names, output_op_name

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def test(net, pa, emptyArray):
    # 100% resource occupation - with 5 pending big jobs (20 tu) *(10,10)
    num_of_new_jobs = pa.num_nw
    nw_len_seqs = np.zeros((1, 5), dtype='int32')
    nw_len_seqs.fill(20)
    nw_size_seq = np.zeros((1, num_of_new_jobs, 2), dtype='int32')
    nw_size_seq.fill(10)

    env = Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seq,
              render=False, end="all_done")

    for i in range(num_of_new_jobs):
        env.step(5)
    print(("Job queue occupation is {}/5\nJob backlog occupation is {}/60 ").format(
        len(env.job_slot.slot), env.job_backlog.curr_size))
    jobLog = env.observe()
    env.plot_state()

    resource_cols = np.concatenate(
        [emptyArray[:, 0:10], emptyArray[:, 60:70]]).ravel()
    varNumArr = np.concatenate(emptyArray).ravel()
    LowerBoundList = np.concatenate(jobLog).ravel()
    UpperBoundList = np.concatenate(jobLog).ravel()
    job_cols = np.concatenate(
        [emptyArray[:, 10:60], emptyArray[:, 70:120]]).ravel()

    for i in np.nditer(resource_cols):
        LowerBoundList[i] = 0.1
    for i in np.nditer(resource_cols):
        UpperBoundList[i] = 1
    for var in zip(varNumArr, LowerBoundList, UpperBoundList):
        net.setLowerBound(var[0], var[1])
        net.setUpperBound(var[0], var[2])

    for outputVar, i in enumerate(net.outputVars[0][0:5]):
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
        eq.addAddend(-1, net.outputVars[0][5])
        eq.addAddend(1, i)
        eq.setScalar(0)
        net.addEquation(eq)

    return jobLog


test_number = "4"

pa = Parameters()

pb_file_name = "model/nnet_conv_1820.pb"
net, _, _ = create_network(pb_file_name)

emptyArray = np.arange(2480, dtype='int32').reshape((20, 124))
test(net, pa, emptyArray)
print("Run marabou solver")
vals, stats = net.solve()

print("marabou run result: {} ".format(
    'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))
if len(list(vals.items())) != 0:
    print("\n*********** \nTF outputs from marabou:")
    outputs = [vals[net.outputVars.item(i)]
               for i in range(net.outputVars.size)]
    softmaxed_outputs = softmax(outputs)

    for i in range(net.outputVars.size):
        print("output {} = {} |||| after softmax:  {:.5}".format(
            i, outputs[i], softmaxed_outputs[i]))
    print("max value: {}".format(
        np.argmax(outputs)))
    print("*********************\n")
    print("Run stats:")
    print("getTotalTime: {}".format(stats.getTotalTime()))
    print("getMaxStackDepth: {}".format(stats.getMaxStackDepth()))
    print("getNumPops: {}".format(stats.getNumPops()))
    print("getNumVisitedTreeStates: {}".format(
        stats.getNumVisitedTreeStates()))
    print("getNumTableauPivots: {}".format(stats.getNumTableauPivots()))
    print("getMaxDegradation: {}".format(stats.getMaxDegradation()))
    print("getNumPrecisionRestoratins: {}".format(
        stats.getNumPrecisionRestorations()))
    print("getNumSplits: {}".format(stats.getNumSplits()))
    print("getNumPrecisionRestorations: {}".format(
        stats.getNumPrecisionRestorations()))
    print("getNumSimplexPivotSelectionsIgnoredForStability: {}".format(
        stats.getNumSimplexPivotSelectionsIgnoredForStability()))
    print("getNumSimplexUnstablePivots: {}".format(
        stats.getNumSimplexUnstablePivots()))
print("end")

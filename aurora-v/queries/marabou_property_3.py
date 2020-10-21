
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from tensorflow.python.saved_model import tag_constants
import utils



def create_network(filename):
    output_op_name = "model/pi/add"
    input_op_names = ["input/Ob"]
    network = Marabou.read_tf(filename, inputName=input_op_names,outputName=output_op_name) #,savedModel = True,outputName = "save_1/restore_all", savedModelTags=[tag_constants.SERVING] )
    return network, input_op_names, output_op_name

# Network inputs:
# 0 - 9   : latency gradient, the derivative of latency with respect to time
# 10 - 19 : latency ratio, the ratio of the current MI’s mean latency to minimum observed mean latency of any MI in
#           the connection’s history
# 20 - 29 : sending ratio, the ratio of packets sent to packets acknowledged by the receiver

def basic_test(filename, to_log_file=False):

    network,input_op_names, output_op_name =  create_network(filename)

    # Get the input and output variable numbers; [0] since first dimension is batch size
    inputVars = network.inputVars[0][0]

    outputVars = network.outputVars[0]
    print("inputVars len =", len(inputVars))
    print("outputVars len =", len(outputVars))
    print("outputVars =", outputVars)
    print("network outputVars =", network.outputVars)
    print("outputVars[0]  =", outputVars[0])
    print("outputVars[0].type  =", type(outputVars[0]))
    print(network.inputVars)

    sanity_inputs =[]
    eps0 = network.getNewVariable()
    eps1 = network.getNewVariable()

    network.setLowerBound(eps0, -0.01)
    network.setUpperBound(eps0, 0.01)
    network.setLowerBound(eps1, 0)
    network.setUpperBound(eps1, 0.01)

    latency_gradient_indices = [i for i in range (0,len(inputVars),3)]
    latency_ratio_indices = [i+1 for i in range (0,len(inputVars),3)]
    sending_ratio_indices = [i+2 for i in range (0,len(inputVars),3)]


    for i in latency_gradient_indices:
        sanity_inputs.append(0)
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
        eq.addAddend(1, inputVars[i])
        eq.addAddend(-1, eps0)
        eq.setScalar(0)
        network.addEquation(eq)

    for i in latency_ratio_indices:
        sanity_inputs.append(1)
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
        eq.addAddend(1, inputVars[i])
        eq.addAddend(-1, eps1)
        eq.setScalar(1)
        network.addEquation(eq)

    for i in sending_ratio_indices:
        l = 2
        u = 20
        network.setUpperBound(inputVars[i], u)
        network.setLowerBound(inputVars[i], l)
        sanity_inputs.append((u+l)//2)

    for i in range(len(outputVars)):
        network.setLowerBound(outputVars[i], 0)
        network.setUpperBound(outputVars[i], 100)

    query_info = "-0.01<=latency_gradient<= 0.01, 1<=latency_ratio_indices<=1.01, sending_ratio_indices >= 2" \
                 "\noutput >= 0 "
    print("\nMarabou results:\n")

    # Call to C++ Marabou solver
    if to_log_file:
        vals, stats = network.solve("results/vrl_marabou.log",verbose=False)
        print('marabou solve run result: {} '.format(
            'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))
    else:
        vals, stats = network.solve(verbose=True)
        print(vals)
        print('marabou solve run result: {} '.format(
            'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))

    # utils.write_results_to_file(vals,inputVars, outputVars, "query3",query_info,".")

import sys

def main():

    if len(sys.argv) < 2:
        print("usage:",sys.argv[0], "<pb_filename> ")
        exit(0)

    filename = sys.argv[1]
    basic_test(filename )



if __name__ == "__main__":
    main()

import sys
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from tensorflow.python.saved_model import tag_constants
import utils


def create_network(filename,k):
    output_op_name = "model/pi/add"

    input_op_names = ["input/Ob"]
    network = Marabou.read_tf_k_steps(filename, k,inputName=input_op_names,outputName=output_op_name)
    return network, input_op_names, output_op_name

# Network inputs:
# 0 - 9   : latency gradient, the derivative of latency with respect to time
# 10 - 19 : latency ratio, the ratio of the current MI’s mean latency to minimum observed mean latency of any MI in
#           the connection’s history
# 20 - 29 : sending ratio, the ratio of packets sent to packets acknowledged by the receiver

def k_test(filename,k, to_log_file=False):
    network, input_op_names, output_op_name = create_network(filename,k)



    outputVars = network.outputVars
    # print("outputVars =", outputVars)
    # print("outputVars len =", len(outputVars))

    assert (len(outputVars) == k)
    # print("network outputVars =", network.outputVars)

    # epsilon for bounding latency gradient (for each k)
    latency_gradient_eps = []
    for i in range(k):
        eps = network.getNewVariable()
        print()
        network.setLowerBound(eps, -0.01)
        network.setUpperBound(eps, 0.01)
        latency_gradient_eps.append(eps)

    # epsilon for bounding latency ratio
    latency_ratio_eps = network.getNewVariable()
    network.userDefineInputVars.append(latency_ratio_eps)

    network.setLowerBound(latency_ratio_eps, 1)
    network.setUpperBound(latency_ratio_eps, 1.01)


    # latency gradient new inputs
    new_inputs = []
    for i in range(k):
        new_intput = network.getNewVariable()
        network.userDefineInputVars.append(new_intput)

        print("new input var = ", new_intput)
        new_inputs.append(new_intput)
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
        eq.addAddend(-1, latency_gradient_eps[i])
        eq.addAddend(1, new_intput)
        eq.setScalar(0)
        network.addEquation(eq)

    b = 0
    for j in range(k):
        inputVars = network.inputVars[j][0]
        latency_gradient_indices = [i for i in range(0, len(inputVars), 3)]
        latency_ratio_indices = [i + 1 for i in range(0, len(inputVars), 3)]
        sending_ratio_indices = [i + 2 for i in range(0, len(inputVars), 3)]
        # print("latency_gradient_indices:")
        for i in latency_gradient_indices:
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            new_input_idx = (b+j)%(k)
            eq.addAddend(-1, inputVars[i])
            eq.addAddend(1, new_inputs[new_input_idx])

            eq.setScalar(0)
            network.addEquation(eq)
            b+=1

        for i in latency_ratio_indices:
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            # network.userDefineInputVars.append(inputVars[i])
            eq.addAddend(1, inputVars[i])
            eq.addAddend(-1, latency_ratio_eps)
            eq.setScalar(0)
            network.addEquation(eq)

        for i in sending_ratio_indices:
            l = 1 # No loss
            u = 1
            network.userDefineInputVars.append(inputVars[i])
            network.setUpperBound(inputVars[i], u)
            network.setLowerBound(inputVars[i], l)

    for i in range(len(outputVars)):
        network.setLowerBound(outputVars[i], -0.001)
        network.setUpperBound(outputVars[i], 0.001)

    query_info = "-0.01<=latency_gradient<= 0.01, 1<=latency_ratio_indices<=1.01, sending_ratio_indices = 1\n" \
                 "output=0 (with a little error)"

    print("\nMarabou results:\n")
    # network.saveQuery(results/basic_query")


    vals, stats = network.solve()#"results/vrl_marabou.log",verbose=False)#, options = options)
    print(vals)
    result = 'SAT' if len(list(vals.items())) != 0 else 'UNSAT'
    print('marabou solve run result: {} '.format(
        result))
    # utils.write_results_to_file(vals,inputVars, outputVars, "query1",query_info,".",k)
    return result


import sys

def main():

    if len(sys.argv) not in [3]:
        print("usage:",sys.argv[0], "<pb_filename> [k] ")
        exit(0)
    filename = sys.argv[1]
    k = int(sys.argv[2])
    k_test(filename,k,False)

if __name__ == "__main__":
    main()

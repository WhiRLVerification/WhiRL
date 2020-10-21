
import os

# will write to dir/results/query_name.rslt
def write_results_to_file (vals, input_vars , output_vars, query_name, info = " ",dir = ".",k = 0):

    if k>0:
        query_name+=("_k"+str(k))
    path = dir+"/results/"+query_name+".rslt"

    with open(path, 'w') as out:
        out.write(" _____  ______  _____ _    _ _   _______ _____"+os.linesep)
        out.write("|  __ \|  ____|/ ____| |  | | | |__   __/ ____|"+os.linesep)
        out.write("| |__) | |__  | (___ | |  | | |    | | | (___"+os.linesep)
        out.write("|  _  /|  __|  \___ \| |  | | |    | |  \___ \\"+os.linesep)
        out.write("| | \ \| |____ ____) | |__| | |____| |  ____) |"+os.linesep)
        out.write("|_|  \_\______|_____/ \____/|______|_| |_____/"+os.linesep)
        out.write("============== "+query_name+" =============="+os.linesep)

        out.write(info+os.linesep)
        result = 'SAT' if len(list(vals.items())) != 0 else 'UNSAT'
        out.write('marabou solve run result: {} '.format(result
            )+os.linesep)
        if result == 'UNSAT': return 
        for i in input_vars:
            out.write("input var #"+str(i)+" = "+str(vals[i])+os.linesep)

        for i in output_vars:
            out.write("output var #"+str(i)+" = "+str(vals[i])+os.linesep)
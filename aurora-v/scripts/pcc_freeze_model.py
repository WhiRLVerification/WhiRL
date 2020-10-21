import tensorflow as tf

from tensorflow.python.framework import graph_io
def frozen_graph_maker(export_dir,output_graph):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        output_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        gd = sess.graph.as_graph_def()
	# fix nodes
        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                if 'validate_shape' in node.attr: del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    node.input[0] = node.input[1]
                    del node.input[1]

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            gd,
            output_nodes# The output node names are used to select the usefull nodes
            )       
        # Finally we serialize and dump the output graph to the filesystem
        # print(output_nodes[-1])
	
	
        graph_io.write_graph(output_graph_def, output_graph, 'output_graph.pb', as_text=False)
        # with tf.gfile.GFile(output_graph, "wb") as f:
        #     f.write(output_graph_def.SerializeToString())


import sys

def main():
    if len(sys.argv)!=3:
        print("usage:",sys.argv[0], "<export_dir> <output_graph_dir>")
        exit(0)
    export_dir = sys.argv[1]
    output_graph = sys.argv[2] #"frozen_graph_2.pb"
    frozen_graph_maker(export_dir, output_graph)


if __name__ == "__main__":
    main()


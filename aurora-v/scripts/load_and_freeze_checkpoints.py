import tensorflow as tf
import os
from tensorflow.python.framework import graph_io


def frozen_graph_maker(checkpoints_dir,model_name, output_graph):
    # with tf.Session(graph=tf.Graph()) as sess:
    #     tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    #     output_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #     gd = sess.graph.as_graph_def()
    idx = ""
    #checkpoint_file_format = '{}/{}_{}.{}'
    checkpoint_file_format = '{}/{}{}.{}'
    #.format(checkpoints_dir,model_name,idx,suffix) # checkpoints_dir+"/"+model_name++str(0)+
    while os.path.isfile(checkpoint_file_format.format(checkpoints_dir,model_name,idx,"ckpt.meta")):
        print("Yep")
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(checkpoint_file_format.format(checkpoints_dir,model_name,idx,"ckpt.meta")) #/tmp/model.ckpt.meta')
            saver.restore(sess, checkpoint_file_format.format(checkpoints_dir,model_name,idx,"ckpt"))
            output_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
            gd = sess.graph.as_graph_def()
            # fix nodes
            for node in gd.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
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
                sess,  # The session is used to retrieve the weights
                gd,
                output_nodes  # The output node names are used to select the usefull nodes
            )
            # Finally we serialize and dump the output graph to the filesystem
            # print(output_nodes[-1])

            graph_io.write_graph(output_graph_def, output_graph, 'output_graph_{}.pb'.format(idx), as_text=False)
            # with tf.gfile.GFile(output_graph, "wb") as f:
            #     f.write(output_graph_def.SerializeToString())
            #idx+=1
            break

import sys


def main():
    if len(sys.argv) != 4:
        print("usage:", sys.argv[0], "<checkpoints_dir> <model_name> <output_graphs_dir>")
        exit(0)
    checkpoints_dir = sys.argv[1]
    model_name = sys.argv[2]
    output_graph = sys.argv[3]
    frozen_graph_maker(checkpoints_dir,model_name, output_graph)


if __name__ == "__main__":
    main()


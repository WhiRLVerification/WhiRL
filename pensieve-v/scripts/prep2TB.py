import tensorflow as tf
from tensorflow.python.platform import gfile


def import_pb_to_tensorboard(pbFile,logdir):
    with tf.Session() as sess:
        model_filename = pbFile
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
    train_writer = tf.summary.FileWriter(logdir)
    train_writer.add_graph(sess.graph)
    train_writer.flush()
    train_writer.close()



import sys

def main():
    if len(sys.argv)!=3:
        print("usage:",sys.argv[0], "<pb_file> <logs_dir>")
        exit(0)
    pbFile  = sys.argv[1]
    logdir = sys.argv[2]
    import_pb_to_tensorboard(pbFile, logdir)


if __name__ == "__main__":
    main()

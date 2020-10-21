
import numpy as np
import os
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import graph_util
import tensorflow as tf


def genShapeMap(ops):
    """
    Function to set input operations
    Arguments:
        [ops]: (tf.op) list representing input
    """
    shapeMap = dict()
    for op in ops:
        try:
            shape = tuple(op.outputs[0].shape.as_list())
            shapeMap[op] = shape
        except:
            shapeMap[op] = [None]
    return shapeMap

def evaluateNetwork( pbFileName ,inputValues,inputNames, outputName):
    """
    Function to evaluate network at a given point using Tensorflow
    Arguments:
        inputValues: list of (np array)s representing inputs to network
    Returns:
        outputValues: (np array) representing output of network
    """

    with tf.gfile.GFile(pbFileName, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    mySess = tf.Session(graph=graph)
    inputOps = []
    for i in inputNames:
        inputOps.append(mySess.graph.get_operation_by_name(i))
    shapeMap = genShapeMap(inputOps)
    print("Evaluating network")
    inputValuesReshaped = []
    for j in range(len(inputOps)):
        inputOp = inputOps[j]
        inputShape = shapeMap[inputOp]
        inputShape = [i if i is not None else 1 for i in inputShape]
        # Try to reshape given input to correct shape
        inputValuesReshaped.append(inputValues[j].reshape(inputShape))

    inputNames = [o.name + ":0" for o in inputOps]
    feed_dict = dict(zip(inputNames, inputValuesReshaped))
    out = mySess.run(outputName + ":0", feed_dict=feed_dict)

    return out[0]
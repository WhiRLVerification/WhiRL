# WhiRL

general tool for verifying deep-RL-driven systems

# Overview
WhiRL is designed to allow a user to provide an RL-produced DNN and a desirable safety or liveness property, and then to seamlessly
produce from these inputs BMC queries.

user is required to provide:

- the actual DNN controller, given in TensorFlow format;
- the state space of the system, which is typically the same as the inputs to the DNN controller;
- a definition of the initial states;
- the transition relation,which specifies how the system reacts once the DNN controller makes a selection;
- a predicate defining the bad states (for safety), or a predicate defining the good states (for liveness); 
- the parameters K and L for the BMC queries.
The framework can also be configured to increment K in each iteration, until a violation is found or until it runs out of resources.


This repository contains the main Python module, an interface to the Marabou framework, which reads a deep-RL DNN
in TensorFlow format and allows user to specify safely, liveness and bounded liveness properties to be verified over the DNN. The repository also contains scripts for running our case study
experiments - Pensieve and Aurora.

# Getting Started

Install the Marabou framework (https://github.com/NeuralNetworkVerification/Marabou), with -DBUILD_PYTHON=ON (see 'Python API' section in the Marabou repo).
Add and replace the files from WhiRL/maraboupy with the files in Marabou/maraboupy. (cp maraboupy/* Marabou/maraboupy/)
for examples, run the the Aurora/Pensieve experiments. (cd [EXPERIMENT]; python3 queries/[QUERY] model/output_graph.pb [K] [others]). 
additional scripts are availalbe.


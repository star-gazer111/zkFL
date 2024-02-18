# zkFL

zkFL stands for Zero-Knowledge based Differentially Private Federated learning. It is a unique way to train DL models using decentralized computing & Zero-Knowledge proofs for enhanced security & faster computations based on trustlessness & ultra-privacy.

Federated Learning is a privacy-preserving scheme to train deep learning models. Data exists in isolated pools and clients that are part of the network train a model with base parameters on their data. They share the updated model parameters with an aggregator that takes the federated average of this set of models. The result is going to be a new updated base model for the next epoch of training.

To remove the dependency on the server, we leverage ZK-Proofs to make the server trustless. The Zk-Proofs are then shown publicly so that anyone can verify whether or not the computation was done correctly. 

## How it's made

The foundation of this idea is this research paper on Zero-Knowledge Federated Learning

How it works

It takes the gradient parameters from the clients taking part in the learning process, aggregates it trustlessly using ZK, and then sends the updated Params back to the clients. In the end, we have a differentially private distributed learning system with a trustless server. 

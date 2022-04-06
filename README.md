# DACPGTN
An end-to-end model DACPGTN for predicting ATC codes for a given drug. DACPGTN constructs composite features of drugs, diseases and targets by applying multiple biomedical information, and generates new node heterogeneous networks from known heterogeneous networks based on Graph Transformer Network. Finally, a graph convolutional network generates embeddings of drug nodes, which are further used for drug discovery tasks in multi-label learning.

# Requirements
* python == 3.6
* pytorch == 1.5.1

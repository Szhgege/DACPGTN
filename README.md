# DACPGTN
An end-to-end model DACPGTN for predicting ATC code for a given drug. DACPGTN constructs composite features of drugs, diseases and targets by applying multiple biomedical information, and generates new node heterogeneous networks from known heterogeneous networks based on Graph Transformer Network. Finally, a graph convolutional network generates embeddings of drug nodes, which are further used for drug discovery tasks in multi-label learning.
![image](https://github.com/Szhgege/DACPGTN/blob/main/data/framework.jpg)
Overall framework of DACPGTN. The feature information of different biomedical entities is integrated to construct a composite feature matrix as the node feature input of the prediction module (PartA). The graph transformer layer is used to obtain the potential interactions information between different biomedical entities from heterogeneous networks set (Part B). The prediction stage uses the composite feature matrix and the learned potential interaction information networks to obtain prediction results (PartC).
# Requirements
* python == 3.6
* pytorch == 1.5.1
* scikit-learn == 0.23.2
* Keras == 2.3.1
* numpy == 1.19.2

If this work is useful for your research, please cite this article https://www.frontiersin.org/articles/10.3389/fphar.2022.907676/full.






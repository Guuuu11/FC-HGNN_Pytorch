# FC-HGNN:A Heterogeneous Graph Neural Network Based on Brain Functional Connectivity for Mental Disorder Identification

This is a PyTorch version of FC-HGNN model as proposed in our paper. We're in the process of organizing the code and will share it publicly soon!


## Introduction
Rapid and accurate diagnosis of mental disorders has long been an essential challenge in clinical medicine. Due to the advantage in addressing non-Euclidean structures, graph neural networks have been increasingly used to study brain networks. Among the existing methods, the population graph models have achieved high predictive accuracy by considering intersubject relationships but weak interpretability limits its clinical applicability. The individual graph approach models functional brain networks and identifies abnormal brain regions that cause diseases but has poor accuracy. To address these issues, we propose a heterogeneous graph neural network based on brain functional connectivity (FC-HGNN), which is an end-to-end model with a two-stage process. In the first phase, the brain connectomic graph is used to extract individual brain features. An integrated intrahemispheric and interhemispheric convolutional graph layer is used to learn brain region features, and a local-global dual-channel pooling layer is used to identify biomarkers. In the second stage, a heterogeneous population graph is constructed based on sex and the fusion of imaging and non-imaging data from subjects.The feature embeddings of same-sex and opposite-sex neighbours are learned separately according to a hierarchical feature aggregation approach. Subsequently, they are adaptively fused to generate the final node embedding, which is then utilized for obtaining classification predictions. The cross-validation and transduction learning results show that FC-HGNN achieves state-of-the-art performance in classification prediction experiments using two public datasets. Moreover, FC-HGNN identifies crucial biomarker regions relevant for disease classification, aligning with existing studies and exhibiting outstanding predictive performance on actual clinical data.

For more details about FC-HGNN, please refer to our paper [[INFFUS](https://www.sciencedirect.com/science/article/pii/S156625352400397X)] 

## Instructions
The public datasets [[ABIDE]( http://preprocessed-connectomes-project.org/abide/)] and [[Rest-meta-MDD](http://rfmri.org/maps)] used in the paper are both downloaded from the official website. Running `main.py` trains the model and makes predictions. `main_transductive.py` adds a validation set at run time. When training and testing your own data, it is recommended to try adjusting the relevant hyperparameters.
## Requirements:
* torch
* torch_geometric
* scipy
* numpy
* os

## Citation

```
@article{gu2024fc,
  title={FC-HGNN: A heterogeneous graph neural network based on brain functional connectivity for mental disorder identification},
  author={Gu, Yuheng and Peng, Shoubo and Li, Yaqin and Gao, Linlin and Dong, Yihong},
  journal={Information Fusion},
  pages={102619},
  year={2024},
  publisher={Elsevier}
}
```

## Postscript
I have traversed a profoundly meaningful and enchanting journey in the realm of scientific research. As I look back on this experience, I extend my heartfelt wishes for your happiness and success in all your endeavors. With sincere regards, a master's student from Ningbo University.
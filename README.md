
Disclaimer: This is not an official Google product.

Copyright 2017 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## SVCCA for Deep Learning Dynamics and Interpretability

This repository contains the core code for implementing Canonical Correlation Analysis on deep neural network representations, which was used for the results in the paper:

[Maithra Raghu](http://maithraraghu.com/), [Justin Gilmer](https://scholar.google.com/citations?user=Ml_vQ8MAAAAJ&hl=en), [Jason Yosinski](http://yosinski.com/), [Jascha Sohl-Dickstein](http://www.sohldickstein.com/) (2017).
["SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability"](https://arxiv.org/abs/1706.05806). Neural Information Processing Systems (NIPS) 2017.


### Dependencies
This code was written in Python 2.7 and relies on the numpy and pandas modules

### Usage
The main function to compute CCA similarities between two representations is the `get_cca_similarity` function in [cca_core.py](cca_core.py). This takes in two arrays of shape (num neurons1, num datapoints) , (num neurons2, num_datapoints),
and outputs pairs of aligned directions, how well aligned they are, as well as coefficients to transform from the original neuron basis to the aligned directions.

The [dft_ccas.py](dft_ccas.py) module builds on this functionality to work with convolutional networks. Convolutional layers have a more natural basis in terms of the number of channels, not raw neurons. Using some of the mathematical properties
of the Discrete Fourier Transform, we have a computationally efficient method for comparing CCA similarity for convolutional layers. See Section 3 in the [paper](https://arxiv.org/pdf/1706.05806.pdf) for more details. The `fourier_ccas` function in
[dft_ccas.py](dft_ccas.py) implements exactly this, taking the raw convolutional activations (num datapoints, height1, width1, num channels1), (num datapoints, height2, width2, num channels2)

Note that according to the theory, we get an exact result if the datapoints used to _generate_ (not train) the activations are translation invariant (any 2d translation of a datapoint x is also in the set of datapoints). But even without this, we can get 
very good results (see Examples section).


### Examples
In the paper, we apply this method to understand several aspects of neural network representations, and we give a couple of examples below.
* Sensitivity to different classes: we compare the CCA similarity of an output neuron corresponding to a particular class with the representations learned in intermediate layers on the Imagenet Resnet. We find that CCA similarity can distinguish between visually different classes (firetruck and dog breeds in the image below) and also show similarities between visually similar classes (husky and eskimo dog, two types of terriers). 
<p align="center">
    <img src="examples/Imagenet_class_similarity.png" width=700px>
</p>

* Learning Dynamics: we also compare all pairs of layers in a neural network across time, ending up with pane plots showing how the _representational similarity_ of different layers evolves with time. We find evidence of a _bottom up_ convergence pattern. Layers closer to the input solidify to their final representations first, before layers higher up in the network. This comparison method also highlights other structure properties of the architecture. We see (below top row) that 2x2 blocks are caused by batch norm layers, which are representationally identical to the previous layer. We also see that residual layers (bottom row) result in grid like patterns, having higher representational similarity with previous layers.
<p align="center">
    <img src="examples/dynamics_plots_crop.png" width=700px>
</p>






## Tutorials on (SV)CCA
This folder contains Juypter notebook tutorials on applying CCA to neural network representations. It also overviews related methods, and discusses existing results and open questions. Below is an outline of the notebooks and concepts discussed:
1. [Introduction](https://github.com/google/svcca/blob/master/tutorials/001_Introduction.ipynb): overviews CCA, computing CCA correlation coefficients from neural networks, and how to aggregate the value into a single similarity measurement. 
2. [CCA for Conv Layers](https://github.com/google/svcca/blob/master/tutorials/002_CCA_for_Convolutional_Layers.ipynb): shows how to adapt applying CCA to conv layers, which has multiple different methods
3. [Other Methods, Network Compression](https://github.com/google/svcca/blob/master/tutorials/003_Other_Methods%20_Neuron_Ablations_and_Network_Compression.ipynb) Describes other methods similar to CCA, and using all of these methods to do (1) dimensionality reduction (2) low rank compression of neural network representations
4. [Future Work](https://github.com/google/svcca/blob/master/tutorials/004_Future_Work_and_Open_Questions.ipynb) List of some interesting open questions.

### References

If you use this code, please consider citing either or both of the following papers:

    @incollection{NIPS2017_7188,
    title = {SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability},
    author = {Raghu, Maithra and Gilmer, Justin and Yosinski, Jason and Sohl-Dickstein, Jascha},
    booktitle = {Advances in Neural Information Processing Systems 30},
    editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
    pages = {6076--6085},
    year = {2017},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability.pdf}
    }

<!-- comment to break blocks -->

    @incollection{NIPS2018_7815,
    title = {Insights on representational similarity in neural networks with canonical correlation},
    author = {Morcos, Ari and Raghu, Maithra and Bengio, Samy},
    booktitle = {Advances in Neural Information Processing Systems 31},
    editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
    pages = {5732--5741},
    year = {2018},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/7815-insights-on-representational-similarity-in-neural-networks-with-canonical-correlation.pdf}
    }

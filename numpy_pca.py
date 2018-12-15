# Copyright 2018 Google Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
A simple implementation of PCA (Principle Component Analysis)
using numpy. Written to apply to neural network activations.

"""

import numpy as np

def pca_core(acts, n_components, compute_dirns=False):
    """ Takes in neuron activations acts and number of components.
    Returns principle components and associated eigenvalues.

    Args:
            acts: numpy array, shape=(num neurons, num datapoints)
            n_components: integer, number of pca components to reduce
                          to

    """

    assert acts.shape[0] < acts.shape[1], ("input must be number of neurons"
                                           "by datapoints")

    # center activations
    means = np.mean(acts, axis=1, keepdims=True)
    cacts = acts - means
    
    # compute PCA using SVD
    U, S, V = np.linalg.svd(cacts, full_matrices=False)
    S = np.abs(S)
    
    return_dict = {}
    return_dict["eigenvals"]  = S[:n_components]
    return_dict["neuron_coefs"] = U.T[:n_components]
    if compute_dirns:
        return_dict["pca_dirns"] = np.dot(U.T[:n_components], cacts) + np.dot(U.T[:n_components], means)

    return return_dict

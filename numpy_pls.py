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
An implementation of PLS (Partial Least Squares)
using numpy. Written to apply to neural network activations.

The setup is very similar to the setup in cca_core.py.

See:
https://arxiv.org/abs/1706.05806
https://arxiv.org/abs/1806.05759
for more background on the methods.

"""

import numpy as np

def get_pls_similarity(acts1, acts2):
  """ 

  This function computes Partial Least Squares between two sets of activations.

  Args:
            acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                   datapoints where entry (i,j) is the output of neuron i on
                   datapoint j.
            acts2: (num_neurons2, data_points) same as above, but (potentially)
                   for a different set of neurons. Note that acts1 and acts2
                   can have different numbers of neurons, but must agree on the
                   number of datapoints

            threshold: float between 0, 1 used to get rid of trailing zeros in
                       the cca correlation coefficients to output more accurate
                       summary statistics of correlations.

            compute_dirns: boolean value determining whether actual cca
                           directions are computed. (For very large neurons and
                           datasets, may be better to compute these on the fly
                           instead of store in memory.)

            verbose: Boolean, whether info about intermediate outputs printed

  Returns:
            return_dict: A dictionary with outputs from the cca computations.
                         Contains neuron coefficients (combinations of neurons
                         that correspond to cca directions), the cca correlation
                         coefficients (how well aligned directions correlate),
                         x and y idxs (for computing cca directions on the fly
                         if compute_dirns=False), and summary statistics. If
                         compute_dirns=True, the cca directions are also
                         computed.
  """

  # assert dimensionality equal
  assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
  # check that acts1, acts2 are transposition
  assert acts1.shape[0] < acts1.shape[1], ("input must be number of neurons"
                                           "by datapoints")
  return_dict = {}

  # compute covariance with numpy function for extra stability
  numx = acts1.shape[0]

  covariance = np.cov(acts1, acts2)
  sigmaxx = covariance[:numx, :numx]
  sigmaxy = covariance[:numx, numx:]
  sigmayx = covariance[numx:, :numx]
  sigmayy = covariance[numx:, numx:]

  # compute Partial Least Squares of cross covariance using
  # SVD. Columns of U are coefficients for acts1, rows of V
  # are coefficients for acts2.
  U, S, V = np.linalg.svd(sigmaxy, full_matrices=False)
  S = np.abs(S)
  
  # compute means 
  neuron_means1 = np.mean(acts1, axis=1, keepdims=True)
  neuron_means2 = np.mean(acts2, axis=1, keepdims=True)
    
  # collect return values
  return_dict = {}
  return_dict["eigenvals"] = S
  return_dict["neuron_coeffs1"] = U.T
  return_dict["neuron_coeffs2"] = V
  
  
  pls_dirns1 = np.dot(U.T, (acts1 - neuron_means1)) + neuron_means1
  pls_dirns2 = np.dot(V, (acts2 - neuron_means2)) + neuron_means2

  return_dict["pls_dirns1"] = pls_dirns1
  return_dict["pls_dirns2"] = pls_dirns2
    

  return return_dict

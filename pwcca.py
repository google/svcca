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
The function for computing projection weightings.

See:
https://arxiv.org/abs/1806.05759
for full details.

"""

import numpy as np
import cca_core

def compute_pwcca(acts1, acts2, epsilon=0.):
    """ Computes projection weighting for weighting CCA coefficients 
    
    Args:
         acts1: 2d numpy array, shaped (neurons, num_datapoints)
	 acts2: 2d numpy array, shaped (neurons, num_datapoints)

    Returns:
	 Original cca coefficient mean and weighted mean

    """
    sresults = cca_core.get_cca_similarity(acts1, acts2, epsilon=epsilon, 
					   compute_dirns=True, compute_coefs=True, verbose=False)
    P1, _ = np.linalg.qr(sresults["cca_dirns1"].T)
    P2, _ = np.linalg.qr(sresults["cca_dirns2"].T)
    weights1 = np.sum(np.abs(np.dot(P1.T, acts1.T)), axis=1)
    weights2 = np.sum(np.abs(np.dot(P2.T, acts2.T)), axis=1)
    weights1 = weights1/np.sum(weights1)
    weights2 = weights2/np.sum(weights2)
    
    return np.sum(weights1*sresults["cca_coef1"]), np.sum(weights2*sresults["cca_coef2"]), \
            np.mean(sresults["cca_coef1"]), np.mean(sresults["cca_coef2"])

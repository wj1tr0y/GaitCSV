# GaitCSV: Causal Intervention for Sparse-View Gait Recognition
The training code is based on [OpenGait](https://github.com/ShiqiYu/OpenGait).

The causal intervention triplet loss is in CIML.py. 
It finds anchor, positive, and negative samples based on view labels.

we denote *ap* from the same view as *ap_same*,

*ap* from different views as *ap_diff*,

*an* from the same view as *an_same*,

*an* from different view as *an_diff*.

We adopt mask indexing to support batch-level triplet construction for a faster training process.
So, the code may be ugly. 

We modify the sampler and dataset to support VxPxK sampling strategy.
The main modification is wrapped by several ########.

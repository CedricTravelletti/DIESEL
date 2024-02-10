.. DIESEL documentation master file, created by
   sphinx-quickstart on Sat Feb 10 14:43:25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DIESEL Documentation
==================================

DIESEL is an open source Python package for **DIstributed EStimation of large EnsembLe covariance matrices**.


It uses distributed arrays (`DASK <https://www.dask.org/>`_) to allow handling very large 
covariance matrices and provides distributed implementations of several covariance estimation algorithms. 


See `this article <https://arxiv.org/abs/2311.12909>`_ for an example of how **DIESEL** can be leveraged to reconstruct the climate of the past on a global scale at high-resolution.

.. toctree::
   :maxdepth: 1

   diesel.covariance
   diesel.estimation
   diesel.sampling
   diesel.plotting
   diesel.gridding
   
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

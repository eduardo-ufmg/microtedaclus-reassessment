"""Clustering models behind one interface.

A model learns a batch, predicts the labels of a batch, with -1 for noise and
for any sample predicted before the model has clusters, and reports the size of
its summary of the stream, such as its number of microclusters.
"""

# Implementation of Memory Network in pyTorch

This repo including implementations of End-to-End Memory Network and Key-Value Memory Network

Dataset used here is WikiMovie. Note that the preprocessing scripts are borrowed from [original repo](https://github.com/facebook/MemNN/tree/master/KVmemnn)

# Code (mainly in pytorch)
* config.py: some global configurations for data dir and training detail
* gen\_dict.py: generate dict
* data\_loader.py: load preprocessed data
* model.py:  model for training similarities and readers
* train\_\*.py: training script for experimental model
* plot.py: plot the graph for report

# Run
* run setup\_processed\_data.sh to get the processed data
* run gen\_dict.py to generate the dictionary file
* run gen\_sim\_data.py to generate the data for training similarity function
* run the train\_mlp.py to train the similarity function
* run gen\_reader\_data.py to generate the data for reader
* run train\_lstm.py, train\_memory\_network.py, train\_kv\_mm.py to train the corresponding model

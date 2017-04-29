# NER-Kannada
Named Entity Recognition for History data in Kannada Language

The Project has three modules

* Crawler  ==> NER-Kannada/Crawler
Module to extract wikipedia articles from web


* CRF Based NER System 
--> Executor.py: This module is to create features for CRF and train the model
--> Pre_processing: Scripts for cleaning the raw data
--> Data: It contains input data, model and output files
--> CRF Templates (Feature Templates): This folder contains the template files used for CRF model training
--> Config.ini Files: These files are to configure the path for Executor.py script

* Neural Network based NER System
--> Pre Processing: Module to process raw data and generate input files for traning neural network
--> NER Models: 1) In - Input Files for Neural Networks
                2) out- Output Files of neural network 
                3) src - Modules for training and prediction using neural networks
                4) trained_model - Neural Network NER model files
--> Word Embedding: Module to generate word embeddings.


We obtained 74.61% accuracy with CRF and 73.54% accuracy with Neural Network 


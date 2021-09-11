# ChIPseq-Analysis

seqmodel.py contains CNN+LSTM model. 
Seqmodel functions: Build a model 
                    Train model 
                    Evaluate results on confusion matrix
                    Classify a sequence
                   
analysistool.py class used to preprocess ChIP and Pu-seq data
Analysistool functions: Threshold origins based on efficiency
                        Cleansing Pu-seq datasets
                        Allignment between origins of different datasets
                        Compute difference in efficiencies between two datasets
                        Label a sequence in the ChIP dataset
                        Create a balanced dataset of binding and non-binding sites from Pu-seq data
                        
ChIPseq sample notebook -> sample code to build and train model.

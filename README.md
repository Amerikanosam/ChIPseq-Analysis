# ChIPseq-Analysis

seqmodel.py contains CNN+LSTM model. <br>
Seqmodel functions:<br> Build a model <br>
                    Train model <br>
                    Evaluate results on confusion matrix<br>
                    Classify a sequence<br>
                   
analysistool.py class used to preprocess ChIP and Pu-seq data<br>
Analysistool functions:<br> Threshold origins based on efficiency<br>
                        Cleansing Pu-seq datasets<br>
                        Allignment between origins of different datasets<br>
                        Compute difference in efficiencies between two datasets<br>
                        Label a sequence in the ChIP dataset<br>
                        Create a balanced dataset of binding and non-binding sites from Pu-seq data<br>
                        
ChIPseq sample notebook -> sample code to build and train model.

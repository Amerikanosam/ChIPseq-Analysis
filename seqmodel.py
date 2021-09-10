
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers import Flatten
from keras.layers import Dropout
import tensorflow as tf
import numpy as np


class seqmodel:
    def __init__(self, inputsize=1000):
        self.inputsize = inputsize
    
    def build_model(self,hiddenunits=1000,denseunits=1000,numkernels=32,slwindow=5,lr=0.001):
        #  initate sequential model
        self.model = Sequential()
        # 1-Dimensional convolution
        self.model.add(Conv1D(filters=numkernels,kernel_size=slwindow, activation='relu'))
        # max pooling layer
        self.model.add(MaxPool1D(pool_size=4))
        # flatten maxpool output to 1D array for LSTM
        self.model.add(Flatten()) 
        # repeat input to create 3-dimensional input
        self.model.add(RepeatVector(1000))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(hiddenunits,go_backwards=True,return_sequences=False))
        self.model.add(Dense(denseunits,activation='relu'))
        self.model.add(Dense(1,activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=lr),metrics = ['accuracy'])
        return self.model
    
    def trainmodel(self,numepochs,batchsize,train,trainlabels,val,vallabels):
        tracker = self.model.fit(x=train,y=trainlabels,epochs=numepochs,batch_size=batchsize,validation_data=(val,vallabels))
        return tracker

    def modeleval(self,valdata,vallabels):
        loss, metric = self.model.evaluate(valdata,vallabels)
        return(loss,metric)
    
    def predict(self,data):
        preds=self.model.predict(data)
        return preds
    
    def  confusionmatrix(self,preds,labels,dsthresh):
        self.truepositives = 0
        self.truenegatives = 0
        self.falsepositives = 0
        self.falsenegatives = 0
        # prediction to label
        preds = np.where(preds>dsthresh,1,0)
        # calculate tp,tn,fp,fn
        for i,pred in enumerate(preds):
            label = labels[i]
            if pred == 1:
                if label == 1:
                    self.truepositives+=1
                else:
                    self.falsepositives+=1
            elif pred == 0:
                if label==0:
                    self.truenegatives+=1
                else:
                    self.falsenegatives+=1
        
        # precision,recall,fscore
        precision = self.truepositives/(self.truepositives+self.falsepositives)
        recall = self.truepositives/(self.truepositives+self.falsenegatives)
        fscore = 2*precision*recall/(precision+recall)
        return (precision, recall, fscore)
    

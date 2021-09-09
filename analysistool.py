import torch
import pandas as pd
import random
import numpy as np


class analysistool:

    def __init__(self,mutdf:pd.DataFrame,wtdf:pd.DataFrame):
        self.mutdf = mutdf
        self.wtdf = wtdf
    
    def  preprocess(self,deldf:pd.DataFrame) ->pd.DataFrame :
        deldf.dropna(inplace=True)
        return deldf

    def thresholding(self,delthresh,wtthresh):
        # threshold only mutated strain
        self.mutdf  = self.mutdf[self.mutdf["efficiency"]>delthresh]
        self.mutdf.reset_index(drop=True,inplace=True)

        self.wtdf = self.wtdf[self.wtdf["efficiency"]<wtthresh]
        self.wtdf.reset_index(drop=True,inplace=True)
        return (self.mutdf,self.wtdf)
    
    # compute the difference in efficiencies
    def compute_defficiency(self,deldf,wilddf):
      # store state of df in method
      
      #set difference in efficiency to 0
      deldf["deff"] = 0

      # long iteration over each origin (O(n2))
      for ind1, origin in deldf.iterrows():
        for ind2, wtorigin in wilddf.iterrows():
          #search if origin del is in wildtype datase
          if origin.x in range(int(wtorigin.xmin)-500,int(wtorigin.xmax)+500):
            #compute change in efficiency
            deldf.loc[ind1,"deff"] =   origin.efficiency - wtorigin.efficiency
      deldf.drop(deldf[deldf.deff==0].index)
      deldf.reset_index(drop=True, inplace=True)
            
      return deldf

    def allignment(self):

        # get index range
        delxind = max(self.mutdf.index)
        # create a copy of wildtype df
        for i in range(delxind):
            # origin deleted at location i
            originx = self.mutdf.iloc[i]

            # origin wild type at location i
            wtoriginx = self.wtdf.iloc[i]
            # previous origin wildtype
            wtprev =self.wtdf.iloc[i - 1]
            # next origin wildtype
            wtnext = self.wtdf.iloc[i + 1]

            # if rif1del origin is not alligned with current wild type origin
            if originx.x not in range(int(wtoriginx.xmin) - 2500, int(wtoriginx.xmax) + 2500):

                # if rif1del origin doesnt fall in previous and following origin
                # must add the rif1del origin to the wild type set with an efficiency of 0
                # 0 efficiency means that the origin is suppresed in the presence of rif1
                if (originx.x not in range(int(wtprev.xmin) - 2500, int(wtprev.xmax) + 2500)) and (
                        originx.x not in range(int(wtnext.xmin) - 2500, int(wtnext.xmax) + 2500)):
                    # if origin occurs before wildtype origin
                    # add origin behind the list
                    if (originx.x < wtoriginx.xmin - 1):
                        indloc = i - 0.5
                        tempdf = pd.DataFrame(
                            {"x": originx.x, "efficiency": 0, "xmin": originx.xmin, "xmax": originx.xmax},
                            index=[indloc])
                        self.wtdf = self.wtdf.append(tempdf, ignore_index=False)
                        self.wtdf = self.wtdf.sort_index().reset_index(drop=True)

                    # if origin occurs after wild type current origin
                    # add origin to wildtype in succession
                    elif (originx.x > wtoriginx.xmax + 1):
                        # current index +0.5 allows squeezing between to current rows
                        indloc = i + 0.5
                        # append dataframe with current origing
                        tempdf = pd.DataFrame(
                            {"x": originx.x, "efficiency": 0, "xmin": originx.xmin, "xmax": originx.xmax},
                            index=[indloc])
                        self.wtdf = self.wtdf.append(tempdf, ignore_index=False)
                        self.wtdf = self.wtdf.sort_index().reset_index(drop=True)
                        # if current origin doesnt occur in wildtype it means wildtype doesnt occur to rif1del
                        # add wt origin to rif1del data as well with 0 efficiency
                        indloc = i - 0.5
                        tempdf = pd.DataFrame(
                            {"x": wtoriginx.x, "efficiency": 0, "xmin": wtoriginx.xmin, "xmax": wtoriginx.xmax},
                            index=[indloc])
                        self.mutdf = self.mutdf.append(tempdf, ignore_index=False)
                        self.mutdf = self.mutdf.sort_index().reset_index(drop=True)

                # if origin occurs in previous origin of wildtype
                # possible since base ranges in wildtype and rif1 origins differ
                # must copy the wildtype once more

                elif (originx.x in range(int(wtprev.xmin) - 2500, int(wtprev.xmax) + 2500)):

                    indloc = i - 0.5
                    tempdfwt = pd.DataFrame(
                        {"x": wtprev.x, "efficiency": wtprev.efficiency, "xmin": wtprev.xmin, "xmax": wtprev.xmax},
                        index=[indloc])
                    self.wtdf = self.wtdf.append(tempdf, ignore_index=False)
                    self.wtdf = self.wtdf.sort_index().reset_index(drop=True)


                elif (originx.x in range(int(wtnext.xmin) - 2500), int(wtnext.xmax) + 2500):

                    indloc = i - 0.5
                    tempdf = pd.DataFrame(
                        {"x": wtoriginx.x, "efficiency": 0, "xmin": wtoriginx.xmin, "xmax": wtoriginx.xmax},
                        index=[indloc])
                    self.mutdf = self.mutdf.append(tempdf, ignore_index=False)
                    self.mutdf = self.mutdf.sort_index().reset_index(drop=True)

        return (self.mutdf,self.wtdf)



    def cutoffde(self, thresh,lowthresh,mutdf,wtdf):
      #mutdf["defficiency"] = mutdf["efficiency"] - wtdf["efficiency"]
      bindingsdeff = mutdf[mutdf.deff>thresh]
      # extract completely suppressed origins
      bindingssup = mutdf[mutdf["deff"]==0]
      bindings = pd.concat([bindingsdeff,bindingssup])
   
      return bindings

    def labelchip(self,chipdf,chrdeldf):
      chipdf["label"] = 0
      for i,origin in chrdeldf.iterrows():
        chipdf.loc[int(origin.x)-2500:int(origin.x)+2500,"label"]=1
      return chipdf

    #  takes chip and thresholded puseq data to create the sequences of bindingsites
    def createdataset(self,thresh,lowthresh,mutdf,wtdf,chipdf):
      bsdataset = []
      nonbsdataset = []
      labels = []
      nonbslabels = []
      bsxset = []
      nonbsxset = []


      #run cutoff difference of efficiency
      bindings = self.cutoffde(thresh,lowthresh,mutdf,wtdf)
      # cast int32 to bindings assist iterations
      bindings = bindings[["x","xmin","xmax"]].astype('int32')

      #cut off telomeres
      #get ends of chromosome
      end1 = chipdf.x.min()
      end2 = chipdf.x.max()
      telomere1 = [end1,end1+19000]
      telomere2 = [end2-19000,end2]
      # remove telomeres from puseq data
      for ind, origin in bindings.iterrows():
        if (origin.x in range(telomere1[0],telomere1[1])) or (origin.x in range(telomere2[0],telomere2[1])):
          bindings.drop(index=ind,inplace=True)
    

      # reset indexes after telomere crop
      bindings.reset_index(drop=True,inplace=True)


      bindingsites= []
      #label all chip as non binding sites


      for ind,origin in bindings.iterrows():
        # create label on chip data
        chipdf.loc[origin.x-2500:origin.x+2499,"label"] =1
        signal = chipdf.loc[origin.x-2500:origin.x+2499,["x","norm"]]
        
        bstensor = torch.tensor(signal["norm"].values,dtype=torch.float32)
        # bin data
        for i in range(0,5000//5):
          bstensor[i] = torch.mean(bstensor[i*5:(i+1)*5])
        bstensor=bstensor[:len(bstensor)//5]
        bsx = torch.tensor(signal["x"].values,dtype=torch.float32)
        bsxset.append(bsx)
        bsdataset.append(bstensor)
        labels.append(1)
        bindingsites.append(origin.x)

      
      non_bs_chip = chipdf[chipdf["label"] == 0]
      # get non binding sites from chip
      non_bs_chip.reset_index(drop=True,inplace=True) 
      
      # randomly acquire non binding sites

      # create nonbinding dataset size of binding sites
      for i in range(len(bsdataset)):
        state=True
        # while loop until correct sequence is acquired
        while state:
          # select random index between 2500 and end of dataframe
          # limit 2500 to acquire 2500 previous positions
          ind = random.randint(5000,len(non_bs_chip))
          signal = non_bs_chip.loc[ind-2500:ind+2499,["x","norm"]]
          # as binding sites have been cropped
          # ensure that nonbinding site positions is a complete fragment
          if (signal.x.max()-signal.x.min()<5100) and (signal.norm.max()<3.5):

            #store read counts in tensor, cast float
            nonbstensor = torch.tensor(signal["norm"].values,dtype=torch.float32)
            # bin data
            for i in range(0,5000//5):
              nonbstensor[i] = torch.mean(nonbstensor[i*5:(i+1)*5])
            nonbstensor = nonbstensor[:len(nonbstensor)//5]
            
            #store positions in tensor, cast float
            nonbsx = torch.tensor(signal["x"].values,dtype=torch.float32)     
            #append list of tensors
            nonbsxset.append(nonbsx)
            nonbsdataset.append(nonbstensor)
            # add label to sequence
            nonbslabels.append(0)       
            state=False
  
      xpos = bsxset + nonbsxset
      dataset = bsdataset + nonbsdataset
      labels = labels + nonbslabels
      return (dataset,labels,xpos,bindingsites)
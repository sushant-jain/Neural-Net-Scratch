import numpy as np
 
data=np.load("predictions.npy")
index=np.arange(len(data))
final=np.empty([len(data),2])
final[:,0]=index
final[:,1]=data
print final.shape
np.savetxt("pred_batch.csv",final,delimiter=',')
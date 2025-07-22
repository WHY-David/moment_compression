import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)  

plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow') 


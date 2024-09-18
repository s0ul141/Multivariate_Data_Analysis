
import numpy as np
import pandas as pd
from sklearn import datasets
from scipy.stats import f

#One Sample Problem
def OneSampleT2Test(X):
 n,p=X.shape
 X0=(6.1, 2.9, 4.5, 1.4)
 delta=np.mean(X,axis=0)-X0
 Sx=np.cov(X,rowvar=False)
 t_squared=n*np.matmul(np.matmul(delta.transpose(),np.linalg.inv(Sx)),delta)
 statistic=t_squared*(n-p)/(p*(n-1))
 F=f(p,n-p-1)
 p_value=1-F.cdf(statistic)
 print(f"Test statistic:{statistic}\n Degrees of freedom:{p} and {n-p}\n p-value:{p_value}\n delta:{delta}")
 return statistic, p_value


iris = pd.read_csv("iris.csv") ## Change Accordingly
#print("shape of dataframe", data.shape)
#Extracting the versicolor with all variables
versicolor_d=iris.loc[iris['Species']== 'Iris-versicolor'] 
#print("shape of dataframe", versicolor.shape)
versicolor=versicolor_d[["SepalLengthCm","SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
OneSampleT2Test(versicolor)

#Two Sample Problem
def TwoSampleT2Test(X,Y):
 nx,p=X.shape
 ny,_=Y.shape
 delta=np.mean(X,axis=0)-np.mean(Y,axis=0)
 Sx=np.cov(X,rowvar=False)
 Sy=np.cov(Y,rowvar=False)
 S_pooled=((nx-1)*Sx+(ny-1) * Sy)/(nx+ny-2)
 t_squared=(nx*ny)/(nx+ny)*np.matmul(np.matmul(delta.transpose(),np.linalg.inv(S_pooled)),delta)
 statistic=t_squared*(nx+ny-p-1)/(p*(nx+ny-2))
 F=f(p,nx+ny-p-1)
 p_value=1-F.cdf(statistic)
 print(f"Test statistic:{statistic}\n Degrees of freedom:{p} and {nx+ny-p-1}\np-value:{p_value}")
 return statistic, p_value


iris = datasets.load_iris()
versicolor = iris.data[iris.target==1, :2]
virginica = iris.data[iris.target==2, :2]
TwoSampleT2Test(versicolor, virginica)




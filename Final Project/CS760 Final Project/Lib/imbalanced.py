import numpy as np


def balance_score(y1,y2):
  index0=np.where(y2==0)[0]
  index1=np.where(y2==1)[0]
  z=y1-y2
  s0=len(np.where(z[index0]==0)[0])/len(index0)
  s1=len(np.where(z[index1]==0)[0])/len(index1)
  return (s0+s1)/2

def upsample(X,y):
  num_sample=len(y)
  num_plus=int(0.3*num_sample)
  index=np.arange(num_sample)
  index1=np.where(y==1)[0]
  index_plus=np.random.choice(index1,num_plus)
  index_result=np.append(index,index_plus)
  np.random.shuffle(index_result)
  X1=X[index_result]
  y1=y[index_result]
  return X1,y1

def undersample(X,y):
  index0=np.where(y==0)[0]
  index1=np.where(y==1)[0]
  num_positive=len(index1)
  num_negative_choose=3*num_positive
  index_negative_choose=np.random.choice(index0,num_negative_choose)
  index_result=np.append(index1,index_negative_choose)
  np.random.shuffle(index_result)
  X1=X[index_result]
  y1=y[index_result]
  return X1,y1
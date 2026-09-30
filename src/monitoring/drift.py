import numpy as np
def psi(expected,actual,bins=10):
 e=np.asarray(expected); a=np.asarray(actual); edges=np.unique(np.quantile(e,np.linspace(0,1,bins+1)))
 if len(edges)<3:return 0.0
 ep,_=np.histogram(e,bins=edges); ap,_=np.histogram(a,bins=edges); ep=ep/len(e)+1e-6; ap=ap/len(a)+1e-6; return float(np.sum((ap-ep)*np.log(ap/ep)))

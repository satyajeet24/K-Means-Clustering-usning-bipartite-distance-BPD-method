import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
from sklearn.datasets import load_iris
import random
"""
from sklearn import datasets
import pandas as pd
style.use('ggplot')

data = pd.read_csv('data_S1.txt')
X= data.iloc[:,:-1].values
"""
X=load_iris().data

no_cluster=4

n_samples, n_features = X.shape
centeroids = np.empty((no_cluster, n_features), dtype=X.dtype)
for i in range(no_cluster):
	center_id = random.randint(1,n_samples)
	for j in range(n_features):
		temp=X[center_id][j]
		centeroids[i][j]=temp

c=[0 for x in range(len(X))]

labels=[0 for i in range(len(X[0]))]
iteration=0

print("old centroid=",centeroids)
while(1):
	lab=[]
	for j in range(len(X)):
		x=[]
		for o in range(len(X[j])):
			x.append(X[j][o])
		dis=[]
		for i in range(len(centeroids)):
			pos=-1
			t=centeroids[i]
			s=0
			ch=[0 for b in range(len(x))]
			for k in range(len(t)):
				d=10**12
				for m in range(len(x)):
					if(ch[m]>=0):
						#print("m=",m)
						d1=float(((t[k]-x[k])**2))
						if(d1<d):
							d=d1
							pos=m
				s+=d
				ch[pos]=-1
			dis.append(s)
		index_min = np.argmin(dis)
		lab.append(index_min)

	for w in range(len(centeroids)):
		for e in range(len(X[0])):
			centeroids[w][e]=0
	for w in range(len(centeroids)):
		s=0
		for h in range(len(lab)):
			if(lab[h]==w):
				s+=1
				for p in range(len(X[h])):
					centeroids[w][p]+=X[h][p]
		if(s>0):
			for f in range(len(X[0])):
				centeroids[w][f]/=s
				az=float(format(centeroids[w][f], '.2f'))
				centeroids[w][f]=az

	iteration+=1
	sm=0
	for i in range(len(lab)):
		if(c[i]==lab[i]):
			sm+=1
	if(sm==len(lab)):
		break
	else:
		for i in range(len(lab)):
			c[i]=lab[i]
	labels=lab
print("New centeroids=",centeroids)
#print("No of iteration= ",iteration)

color = ["g.", "r.", "b.","y.", 'm.', 'k.', 'w.',"c.","g.", "r.", "b.","y.", 'm.', 'k.', 'c.']
#l=['verginica','versicolor','setosca']
for i in range(len(X)):
	#print(i+1,"coordinate: ", X[i], "label:", labels[i])
	plt.plot(X[i][0], X[i][1], color[labels[i]], markersize = 10)
x=[]
y=[]
for i in range(len(centeroids)):
	x.append(centeroids[i][0])
	y.append(centeroids[i][1])
	

plt.scatter(x[0:], y[0:], marker= 'x', s=50, linewidths=5, zorder=10)

plt.show()












import numpy as np
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()

#Making features and labels
features = iris.data[-100:, [0,2]]
labels = (iris.target[-100:]==2).astype(np.int)

#Fitting SVC
from sklearn import svm
from sklearn.svm import SVC

svc = svm.SVC(kernel='linear', C=1000)
svc.fit(features, labels)

#Plotting the graph
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(features[:,0], features[:,1], c=labels, s=30, cmap=plt.cm.Paired)

#Plotting the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

#Creating grid to evaluate model
xx = np.linspace(xlim[0],xlim[1],30)
yy = np.linspace(ylim[0],ylim[1],30)

YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

#Plotting decision boundary
Z = svc.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])

#Plotting SVM
ax.scatter(svc.support_vectors_[:,0], svc.support_vectors_[:,1], s=50, linewidth=1, edgecolor='c')

plt.show()
import os

import numpy as np
import pandas as pd

# Needed for VScode to function correctly, can be deleted for normal operation
os.chdir('/home/cameron/Dropbox/University/PhD/Teaching/COMP219-AI/PythonMachineLearning/chapter02')

import neurons
import plot

df = pd.read_csv('iris.data',
                 header=None)

# Select setosa and versicolor
y = df.iloc[0:150, 4].values
y = np.where(y == 'Iris-virginica', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:150, [1, 3]].values

plot.plotSetosaVersicolor(X)

ppn = neurons.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plot.plotEpochUpdates(ppn)
# plot.plotDecisionRegions(X, y, ppn)

#ada = neurons.AdalineGD(eta=0.1, n_iter=10)
#ada.fit(X, y)
# plot.plotDecisionRegions(X, y, ppn)

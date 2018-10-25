from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np 

import warnings
warnings.filterwarnings("ignore")

def plot_decision_regions(X, y, classifier,                     
                test_idx=None, resolution=0.02):    
    
    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')    
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')    
    cmap = ListedColormap(colors[:len(np.unique(y))])    
    
    # plot the decision surface    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1    
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),       
                      np.arange(x2_min, x2_max, resolution))    
                      
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)    
    Z = Z.reshape(xx1.shape)    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)    
    plt.xlim(xx1.min(), xx1.max())    
    plt.ylim(xx2.min(), xx2.max())    
    
    # plot all samples    
    for idx, cl in enumerate(np.unique(y)):        
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],                    
            alpha=0.8, c=[cmap(idx)],                    
            marker=markers[idx], label=cl)

    # highlight test samples    
    if test_idx:        
        X_test, y_test = X[test_idx, :], y[test_idx]           
        plt.scatter(X_test[:, 0], X_test[:, 1], 
            c='', edgecolor='black', alpha=1.0, 
            linewidth=1, marker='o',                 
            s=100, label='test set')

def print_accuracy(name, y_test, y_pred):
    print('Misclassified samples for {0}: {1}'.format(name, (y_test != y_pred).sum()))
    print('{0} Accuracy: {1:.2f}'.format(name, accuracy_score(y_test, y_pred)))

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

# split into training and test data
X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=0)

# define scaler and scale data
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Define and train perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# Define and train logistic regression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

# Define and train Support Vector Classifier
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

# Define and train our decision tree classifier
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3,
                              random_state=0)
tree.fit(X_train, y_train)

# Define and train our random forest classifier
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)

# Create arrays of names and variables for easier looping
classifierName = ['Perceptron', 'Logistic Regression', 'Support Vector Machine']
classifiers = [ppn, lr, svm]
predictions =[]

# make predictions 
for clf in classifiers:
    predictions.append(clf.predict(X_test_std))

predictions.append(tree.predict(X_test))

for i in range(len(classifierName)):
    print_accuracy(classifierName[i], y_test, predictions[i])

print_accuracy('Random Forest', y_test, predictions[3])

X_combined = np.vstack((X_train, X_test)) # Scaling is not a requirement for decision trees
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt.figure(figsize=(16, 4.8)) # create figure

for i in range(len(classifierName)):
    plt.subplot(1, 5, i + 1) # create a subplot with 4 rows, one column and the i'th member
    plot_decision_regions(X_combined_std, y_combined, classifier=classifiers[i], test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]') 
    plt.ylabel('petal width [standardized]') 
    plt.title(classifierName[i])
    plt.legend(loc='upper left')

# create a subplot on the fourth member for our Random Forest classifier
plt.subplot(1, 5, 4) 
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
plt.title('Decision Tree')

# create a subplot on the fourth member for our Random Forest classifier
plt.subplot(1, 5, 5) 
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.title('Random Forest')

plt.tight_layout()    
plt.show()
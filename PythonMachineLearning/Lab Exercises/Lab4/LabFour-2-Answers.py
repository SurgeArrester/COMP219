import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
import pickle as pk 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from scipy.stats import sem
from sklearn import metrics

faces = fetch_olivetti_faces()

def print_faces(images, target, top_n):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                        hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))
        p.imshow(images[i], cmap=plt.cm.bone)

    # label the image with the target value
    
    plt.show()

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator
    cv = KFold(n_splits=K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)

    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    print("Accuracy on testing set:")
    print(clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

print(faces.images.shape)
print(faces.data.shape)
print(faces.target)

print_faces(faces.images, faces.target, 400)

svc_1 = SVC(kernel='linear')
X_train, X_test, y_train, y_test = train_test_split(faces.data, 
                                                    faces.target, 
                                                    test_size=0.25, 
                                                    random_state=0)

evaluate_cross_validation(svc_1, X_train, y_train, 5)
train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)
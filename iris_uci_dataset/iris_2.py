__output = {}
SHOW_PLTS = True
SHOW_PLTS = False

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print('Train set: {} - Test set: {}'.format(X_train.shape[0], X_test.shape[0]))

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

markers = ("s", "x", "o")
colors = ("red", "blue", "lightgreen")
cmap = ListedColormap(colors[:len(np.unique(y_test))])
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], c=cmap(idx), marker=markers[idx],
                label=cl)
if SHOW_PLTS: plt.show()


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c="", alpha=1.0, linewidth=1, marker="o",
                    s=55, label="test_set")


from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter=40, eta0=0.1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print("Misclassified samples: {}".format((y_test != y_pred).sum()))

from sklearn.metrics import accuracy_score

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,
                      test_idx=range(105, 150))
plt.xlabel("Petal length [standardized]")
plt.xlabel("Petal width [standardized]")
plt.legend(loc="upper left")
plt.show()

from sklearn.svm import SVC

svm = SVC(kernel="rbf", gamma=0.10, C=1.0)
svm.fit(X_train_std, y_train)

print("Train accuracy: {}".format(svm.score(X_train_std, y_train)))
print("Test accuracy: {}".format(svm.score(X_test_std, y_test)))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm,
                      test_idx=range(105, 150))
plt.xlabel("Petal length [standardized]")
plt.xlabel("Petal width [standardized]")
plt.legend(loc="upper left")
plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
knn.fit(X_train_std, y_train)

print("Train accuracy: {}".format(svm.score(X_train_std, y_train)))
print("Test accuracy: {}".format(svm.score(X_test_std, y_test)))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=knn,
                      test_idx=range(105, 150))
plt.xlabel("Petal length [standardized]")
plt.xlabel("Petal width [standardized]")
plt.legend(loc="upper left")
plt.show()

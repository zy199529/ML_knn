from knn import *
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(returnMat[:, 0], returnMat[:, 1], 15.0 * np.array(classLabelVector), 15.0 * np.array(classLabelVector))
plt.show()
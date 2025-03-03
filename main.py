import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.inspection import DecisionBoundaryDisplay

# n_classes = 3
# plot_colors = ["darkblue","mediumvioletred","gold"]
# plot_step = 0.02

# list = []
# for i in range(13):
#     for j in range(i+1, 13):
#         list.append([i,j])
# for pairidx, pair in enumerate(list):
#     # We only take the two corresponding features
#     X = data.data[:, pair]
#     y = data.target

#     # Train
#     clf = DecisionTreeClassifier().fit(X, y)

#     # Plot the decision boundary
#     ax = plt.subplot(9, 9, pairidx + 1)
#     plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
#     DecisionBoundaryDisplay.from_estimator(
#         clf,
#         X,
#         cmap=plt.cm.plasma,
#         response_method="predict",
#         ax=ax,
#         xlabel=data.feature_names[pair[0]],
#         ylabel=data.feature_names[pair[1]],
#     )

#     # Plot the training points
#     for i, color in zip(range(n_classes), plot_colors):
#         idx = np.where(y == i)
#         plt.scatter(
#             X[idx, 0],
#             X[idx, 1],
#             c=color,
#             label=data.target_names[i],
#             edgecolor="black",
#             s=15,
#         )

# plt.suptitle("Decision surface of decision trees trained on pairs of features")
# plt.legend(loc="lower right", borderpad=0, handletextpad=0)
# _ = plt.axis("tight")
# plt.show()

data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
plt.figure()
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
plot_tree(clf, filled=True)
plt.title("Decision tree trained on wine features")
plt.show()


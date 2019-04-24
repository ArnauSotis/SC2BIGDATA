# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:57:27 2019

@author: Arnau
"""

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


                                # Clasificador GAUSSIANO

# Load the digits dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Instantiate a Naive Bayes classifier
clf = GaussianNB()

# Perform 10-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=10)
print("Cross-validation scores: ", cv_scores)

# Print the mean score and the 95% confidence interval of the score estimate
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))


                                # Clasificador KNeighbors (KNN)


### Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

### Tunning k (n_neighbors parameter) using 10-fold cross-validation

k_range = range(1, 31)
k_scores = []

# 1.loop through values of k
for k in k_range:
    # 2. run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())
#print("k-scores: ", k_scores)

# Print the tuned parameter and score
max_idx = k_scores.index(max(k_scores))
print("Best parameter (k_neighbors): {}".format(k_range[max_idx])) 
print("Best score: {0:.2}".format(k_scores[max_idx]))

# Plot the cross-validated accuracy (y-axis) vs the value of K for KNN (x-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Validation curve')

                                #Clasificador KMEANS

    # Load iris dataset
iris = datasets.load_iris()
samples = iris.data

# Visualize the data 
# df = pd.DataFrame(samples, columns=iris.feature_names)
# pd.plotting.scatter_matrix(df, figsize = [8, 8])

# Visualize the data (only petal_length and petal_width)
plt.scatter(samples[:,2],samples[:,3], label='True Position') 
plt.show()

#Create Clusters
model = KMeans(n_clusters=3)
model.fit(samples)
print(model.cluster_centers_) 
print(model.labels_) 

# Visualize how the data has been clustered (only petal_length and petal_width)
plt.scatter(samples[:,2],samples[:,3], c=model.labels_, cmap='rainbow')  
# plot the centroid coordinates of each cluster
plt.scatter(model.cluster_centers_[:,2], model.cluster_centers_[:,3], color='black')
plt.show()

#Cluster labels for new samples
new_samples = [[5.7, 4.4, 1.5, 0.4],[6.5, 3., 5.5, 1.8],  [ 5.8, 2.7, 5.1, 1.9]]
new_labels = model.predict(new_samples)
print(new_labels)

                                    #Clasificador PCA

        
# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())
print(iris.target_names)

# Standardizing the features
sX= StandardScaler().fit_transform(X)
scaledDf = pd.DataFrame(sX, columns=iris.feature_names)
print(scaledDf.head())

# PCA Projection to 2D
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(sX)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
print(principalDf.head())

# Visualize 2D Projection
plt.figure(figsize = (8,8))
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], c=iris.target)
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.title('2 Component PCA', fontsize = 20)
plt.show()

# Explained variance
print(pca.explained_variance_ratio_)
## NAME:SUJITHRA K
## REGISTER NUMBER:212223040212

# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 1. Import the necessary packages using import statement.

 2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

 3.Import KMeans and use for loop to cluster the data.

 4.Predict the cluster and plot data graphs.

 5.Print the outputs and end the program

## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')

print("Dataset Head:\n", df.head())
```

## Output:

![440467315-f6b94c26-9db0-42b3-9ca3-73dd2d0551b2](https://github.com/user-attachments/assets/70753fff-ecba-455b-a60d-c35a5cc3d346)

```
X = df.iloc[:, [3, 4]].values  
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
```
## Output:

![440467848-bf8439e3-5d77-4b6d-980b-c31787a63e0c](https://github.com/user-attachments/assets/197d5160-18b5-48c8-8b39-28c2fc3c6979)

```
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids', edgecolor='black')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
```
## Output:

![440468245-b720aa1b-6960-4613-84c2-6e7e154b7897](https://github.com/user-attachments/assets/e00dfe1e-0fbe-417d-8c0d-4172c75faec2)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

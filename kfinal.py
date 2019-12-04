import numpy as np
import os
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import matplotlib as mpl
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pylab as pl

if __name__ == "__main__":
    #filename = os.path.dirname(__file__) + "/Pre-Survey.csv"
    import pandas as pd
    df=pd.read_csv('Pre-Survey.csv', sep=',')
    #data_points = np.genfromtxt(filename, delimiter=",")
    df1=df [['Title-P','Picture-P','Content-P','Source-P','Authors-P','Date-P']]
    df2=df [['Title-P','Picture-P','Content-P','Source-P','Authors-P','Date-P']]
    df1['Title-P'] = df1['Title-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    df1['Picture-P'] = df1['Picture-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    df1['Content-P'] = df1['Content-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    df1['Source-P'] = df1['Source-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    df1['Authors-P'] = df1['Authors-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    df1['Date-P'] = df1['Date-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    df2['Title-P'] = df2['Title-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    df2['Picture-P'] = df2['Picture-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    df2['Content-P'] = df2['Content-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    df2['Source-P'] = df2['Source-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    df2['Authors-P'] = df2['Authors-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    df2['Date-P'] = df2['Date-P'].apply(lambda x : 0  if x=='Low' else (1 if x=="Moderately low" else (2 if x=="Moderately high" else 3)))
    #df2['Political Leaning'] = df2['Political Leaning'].apply(lambda x : 0  if x=='Liberal' else (1 if x=="Moderate" else (2 if x=="Conservative" else 3))) 
    #df2['Age'] = df2['Age'].apply(lambda x : 0  if x=='18-24' else (1 if x=="25-34" else (2 if x=="35-44" else (3 if x=="45-54" else 4)))) 

    X=df1.to_numpy()
    print(df.head())
    
    pca = PCA(n_components=2).fit(df1)
    pca_2d = pca.transform(df1)
    kmeans = KMeans(n_clusters=3, random_state=111)
    kmeans.fit(df1)
    df2['Cluster'] =  kmeans.labels_
    
    print(kmeans.labels_[0])
    pl.figure('K-means with 6 clusters')
    df2.to_csv (r'clustered_data.csv', index = None, header=True) 
    pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
    #df1['Cluster'] =  kmeans.labels_
    #pl.scatter(df['Source-P'], df['Picture-P'], c=kmeans.labels_)
    """
    print(X[:,0])
    colorsList = ['r','b','g']
    CustomCmap = mpl.colors.ListedColormap(colorsList)
    Clust= ['Cluster1','Cluster2','Cluster3']
    #print(len(X))
    i=0
    
    for i in range(0, df1.shape[0]):
        if(kmeans.labels_[i]==0):
            c1=plt.scatter(X[i, 1], X[i, 3], s=100, c='red',marker='.')
        elif(kmeans.labels_[i]==1):
            c2=plt.scatter(X[i, 1], X[i, 3], s=100, c='blue',marker='.') 
        elif(kmeans.labels_[i]==2):
            c3=plt.scatter(X[i, 1], X[i, 3], s=100, c='green',marker='.') 
           
    #plt.scatter(X[kmeans.labels_==1, 1], X[kmeans.labels_==1, 3], s=50, c='blue', label ='Cluster 2')
    #plt.scatter(X[kmeans.labels_==2, 1], X[kmeans.labels_==2, 3], s=50, c='green', label ='Cluster 3')
    #plt.scatter(X[:,1],X[:,3], c=kmeans.labels_, cmap=CustomCmap)
    red_patch = mpatches.Patch(color='red', label='Cluster1')
    blue_patch = mpatches.Patch(color='blue', label='Cluster2')
    green_patch = mpatches.Patch(color='green', label='Cluster3')
    #plt.legend(handles=[red_patch])
    #plt.legend()
    plt.legend([c1, c2, c3], ['Cluster1', 'Cluster2','Cluster3'])
    #plt.scatter(X[:,0],X[:,1], label='True Position')
    plt.xlabel('Picture')
    plt.ylabel('Source')
    #plt.legend(loc='upper right')
    """
    plt.show()
    
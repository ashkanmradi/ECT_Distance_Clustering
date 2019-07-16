from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from distance import euclidean
import numpy as np
from builtins import len
import networkx as nx
from load import load_data
import kmeansECT.ECT_KMeans as ECT_KMeans
from sklearn import metrics
from sklearn.cluster import KMeans,AgglomerativeClustering

DATASETS = ['iris','ecoli','glass','pima','sonar','wine','soybean','ionosphere','balance','breast']
DATASET_NAME = 'ecoli'

## select which kind of data you want to use. randomly generated data or one of the UCI datasets
random_data = False

if random_data:
    n_samples = 200
    k = 2
    samples, labels = make_circles(n_samples=n_samples, factor=.3, noise=.05)
    bluecircle = samples[labels == 0]
    redcircle = samples[labels == 1]
    all_circles = np.concatenate([bluecircle, redcircle])
    plt.figure()
    plt.scatter(bluecircle[:, 0], bluecircle[:, 1], c='b', marker='o', s=10)
    plt.scatter(redcircle[:, 0], redcircle[:, 1], c='r', marker='+', s=30)
    plt.show()
if random_data == False:
    samples, labels, k = load_data.load_dataset(DATASET_NAME)
    n_samples = len(samples)


nn = np.sqrt(n_samples)
nn = int(nn)
lst = []
distance_matrix = np.zeros((n_samples, n_samples))

## compute euclidean distance between each pair of data
if random_data:
    for i in range(n_samples):
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = euclidean.eclidean_distance(all_circles[i], all_circles[j])
else:
    for i in range(n_samples):
        for j in range(n_samples):
            distance_matrix[i, j] = np.linalg.norm(samples[i] - samples[j])

distance_matrix_copy = distance_matrix.copy()
for i in range(n_samples):
    temp = distance_matrix_copy[i]
    sortedList = sorted(temp)
    max_distance = sortedList[nn]
    for j in range(n_samples):
        if distance_matrix_copy[i, j] > max_distance:
            distance_matrix[i, j] = 0
        else:
            if i!=j and distance_matrix_copy[i, j] != 0:
                distance_matrix[i, j] = 1/distance_matrix_copy[i, j]
            else:
                distance_matrix[i, j] = 0


## convert distance matrix to undirected graph
G = nx.Graph(distance_matrix_copy)

## MST undirected graph
mst_G = nx.minimum_spanning_tree(G)

## convert MST  to matrix
MST_matrix = nx.to_numpy_matrix(mst_G)


## convert distance matrix to directed graph
DG = nx.DiGraph(distance_matrix)

## MST directed graph
mst_dg = nx.algorithms.tree.branchings.Edmonds(DG).find_optimum()

## convert MST_D graph to matrix
MST_matrix_D = nx.to_numpy_matrix(mst_dg)



MST = MST_matrix

for i in range(n_samples):
    for j in range(n_samples):
        if i != j:
            if distance_matrix[i,j] == 0.0 :
                distance_matrix[i,j] = MST[i,j]


distance_matrix_added_MST = distance_matrix

directed_graph = nx.DiGraph(distance_matrix_added_MST)

markov_chain_matrix = np.zeros((n_samples, n_samples))
ai = np.zeros(n_samples)
for i in range(n_samples):
    a_i = np.sum(distance_matrix_added_MST[i])
    ai[i] = a_i
    for j in range(n_samples):
        markov_chain_matrix[i, j] = distance_matrix_added_MST[i, j] / a_i

A = markov_chain_matrix  # adjacency matrix
D = np.diag(ai)
L = D - A



V_G = np.sum(ai)
L_Plus = np.linalg.pinv(L)
ECT = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    e_i = np.zeros(n_samples)
    e_i[i] = 1
    for j in range(n_samples):
        e_j = np.zeros(n_samples)
        e_j[j] = 1
        E = e_i - e_j
        E = E.reshape(n_samples, 1)
        ECT[i, j] = np.sqrt(np.dot(np.dot(np.transpose(E), L_Plus), E) * V_G)





NMI_list_ect_kmeans = {}
NMI_list_kmeans = {}
NMI_list_hierarcy = {}

NUM_ITERATION = 5


sum_nmi_ect_kmeans = 0
sum_nmi_kmeans = 0
sum_nmi_hierarcy = 0

for i in range(NUM_ITERATION):
    print("***************************************************iteration :", i)
    '''---------------- ECT kmeans ---------------'''
    ect_kmeans = ECT_KMeans.ECT_KMeans(samples, k, ECT)
    clusters = ect_kmeans.fit()
    predict_labels_kmeans = ect_kmeans.labels

    NMI_kmeans = metrics.normalized_mutual_info_score(labels, predict_labels_kmeans,average_method='arithmetic')
    sum_nmi_ect_kmeans += NMI_kmeans

    '''---------------- classic kmeans ---------------'''
    k_means = KMeans(n_clusters=k).fit(samples)
    predict_labels_kmeans = k_means.labels_

    NMI_kmeans = metrics.normalized_mutual_info_score(labels, predict_labels_kmeans)
    sum_nmi_kmeans += NMI_kmeans

    '''---------------- Hierarchical ----------------'''
    hierarchical_clus = AgglomerativeClustering(n_clusters=k)
    hierarchical_clus.fit(samples)
    predict_labels_hierarchy = hierarchical_clus.labels_

    NMI_hierarchy = metrics.normalized_mutual_info_score(labels, predict_labels_hierarchy)
    sum_nmi_hierarcy += NMI_hierarchy

NMI_list_ect_kmeans[0] = sum_nmi_ect_kmeans/NUM_ITERATION
NMI_list_kmeans[0] = sum_nmi_kmeans/NUM_ITERATION
NMI_list_hierarcy[0] = sum_nmi_hierarcy/NUM_ITERATION


objects = ('ECT kmeans', 'Classic kmeans', 'Hierarchical')
y_pos = np.arange(len(objects))
performance = [NMI_list_ect_kmeans[0],NMI_list_kmeans[0],NMI_list_hierarcy[0]]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('NMI')
plt.title('Clustering Method')

plt.show()

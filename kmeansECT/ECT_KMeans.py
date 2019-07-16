import random
import sys
import numpy as np


class ECT_KMeans:
    def __init__(self, data, k, ECT):
        self.data = data
        self.k = k
        self.ECT = ECT

    def fit(self, max_iterations=300):
        number_of_clusters = self.k
        prototype_center = {}
        index_list = list(range(len(self.data)))
        random.shuffle(index_list)
        for i in range(number_of_clusters):
            prototype_center[i] = index_list[i]

        print("first prototypes: ", prototype_center)

        J_Error_old = sys.maxsize
        J_Error_best = sys.maxsize
        bestClusterResult = {}

        for iteration in range(max_iterations):

            # create empty clusters
            clusters = {}  # {0:[] , 1:[] , ... , k:[]}
            for i in range(number_of_clusters):
                clusters[i] = set()
                clusters[i].add(prototype_center[i])

            ## Allocation phase
            for x_index in range(len(self.data)):
                ECT_distance = {}
                for item in prototype_center.items():
                    key_center = item[0]
                    center_index = item[1]
                    ECT_distance[key_center] = self.ECT[x_index, center_index] ** 2  # square

                sorted_ECT_distances = sorted(ECT_distance.items(), key=lambda kv: kv[1])
                # add sample to nearest cluster
                key_of_nearest_cluster = sorted_ECT_distances[0][0]

                # if you will index of center or in all_circles , get that with prototype_center[key_of_nearest_cluster]
                clusters[key_of_nearest_cluster].add(x_index)

            # center or prototype update
            for key_c, c in clusters.items():
                listOfSum = {}
                for x_index in range(len(self.data)):  # x_index = x_i
                    _sum = 0
                    for sample_index in c:  # sample_index = x_k
                        ect_dist = self.ECT[sample_index, x_index] ** 2
                        _sum += ect_dist
                    listOfSum[x_index] = _sum
                sorted_sum = sorted(listOfSum.items(), key=lambda kv: kv[1])
                min_x_index = sorted_sum[0][0]
                prototype_center[key_c] = min_x_index

            ## Compute convergence

            J_Error_new = 0
            for key_c, c in clusters.items():
                for sample_index in c:
                    J_Error_new += self.ECT[sample_index, prototype_center[key_c]] ** 2

            if (J_Error_new < J_Error_best):
                J_Error_best = J_Error_new
                bestClusterResult = clusters

            J_Error_old = J_Error_new

            if iteration % 10 == 0:
                print("Iteration: ", iteration, "---  ", J_Error_new, " BEST ERROR: ", J_Error_best)
                print("---------------------")
                print(prototype_center)
                print("---------------------")

        self.labels = [-1] * len(self.data)
        for key, value in clusters.items():
            for i in value:
                print(i,key)
                self.labels[i] = key
        return clusters






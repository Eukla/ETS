import logging
import coloredlogs
import numpy as np
import pandas as pd
from multiprocessing import Pool
import threading
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Sequence, Dict, Optional
from ets.algorithms.early_classifier import EarlyClassifier
import multiprocessing as mp
# Configure the logger (change level to DEBUG for more information)
logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=logger,
                    fmt='%(asctime)s - %(hostname)s - %(name)s[%(process)d] - [%(levelname)s]: %(message)s')

# TODO the algorithm has veery different earliness
x = 0
class ECTS(EarlyClassifier):

    """
    The ECTS algorithm.

    Publications:

    Early classification on time series(2012)
    """

    def __init__(self, timestamps, support: float, relaxed: bool):
        """
        Creates an ECTS instance.

        :param timestamps: a list of timestamps for early predictions
        :param support: minimum support threshold
        :param relaxed: whether we use the Relaxed version or the normal
        """
        self.rnn: Dict[int, Dict[int, List]] = dict()
        self.nn: Dict[int, Dict[int, List]] = dict()
        self.data: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None
        self.mpl: Dict[int, Optional[int]] = dict()
        self.timestamps = timestamps
        self.support = support
        self.clusters: Dict[int, List[int]] = dict()
        self.occur: Dict[int, int] = dict()
        self.relaxed = relaxed
        self.correct: Optional[List[Optional[int]]] = None

    def train(self, train_data: pd.DataFrame, labels: Sequence[int]) -> None:

        """
        Function that trains the model using Agglomerating Hierarchical clustering

        :param train_data: a Dataframe containing-series
        :param labels: a Sequence containing the labels of the data
        """
        self.data = train_data

        self.labels = labels
        if self.relaxed:
            self.__leave_one_out()
        for index, value in self.labels.value_counts().items():
            self.occur[index] = value

        # Finding the RNN of each item
        time_pos = 0
        for e in self.timestamps:
            product = self.__nn_non_cluster(time_pos)  # Changed to timestamps position
            self.rnn[e] = product[1]
            self.nn[e] = product[0]
            time_pos += 1
        temp = {}
        finished = {}  # Dictionaries that signifies if an mpl has been found
        for e in reversed(self.timestamps):
            for index, row in self.data.iterrows():
                if index not in temp:
                    self.mpl[index] = e
                    finished[index] = 0  # Still MPL is not found

                else:
                    if finished[index] == 1:  # MPL has been calculated for this time-series so nothing to do here
                        continue

                    if self.rnn[e][index] is not None:
                        self.rnn[e][index].sort()
                    # Sorting it in order to establish that the RNN is in the same order as the value
                    if temp[index] is not None:
                        temp[index].sort()

                    if self.rnn[e][index] == temp[index]:  # Still going back the timestamps
                        self.mpl[index] = e

                    else:  # Found k-1
                        finished[index] = 1  # MPL has been found!
                temp[index] = self.rnn[e][index]
        self.__mpl_clustering()

    def __leave_one_out(self):
        nn = []
        for index, row in self.data.iterrows():  # Comparing each time-series

            data_copy = self.data.copy()
            data_copy = data_copy.drop(data_copy.index[index])
            for index2, row2 in data_copy.iterrows():
                temp_dist = distance.euclidean(row, row2)

                if not nn:
                    nn = [(self.labels[index2], temp_dist)]
                elif temp_dist >= nn[0][1]:
                    nn = [(self.labels[index2], temp_dist)]
            if nn[0][0] == self.labels[index]:
                if not self.correct:
                    self.correct = [index]
                else:
                    self.correct.append(index)
            nn.clear()

    def __nn_non_cluster(self, prefix: int):
        """Finds the NN of each time_series and stores it in a dictionary

        :param prefix: the prefix with which we will conduct the NN

        :return: two dicts holding the NN and RNN"""
        nn = {}
        rnn = {}
        neigh = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(self.data.iloc[:, 0:prefix + 1])
        def something(row):
            return neigh.kneighbors([row])

        result_data = self.data.iloc[:, 0:prefix + 1].apply(something, axis=1)
        for index, value in result_data.items():
            if index not in nn:
                nn[index] = []
            if index not in rnn:
                rnn[index] = []
            for item in value[1][0]:
                if item != index:
                    nn[index].append(item)
                    if item not in rnn:
                        rnn[item] = [index]
                    else:
                        rnn[item].append(index)
        return nn, rnn

    def __cluster_distance(self, cluster_a: Sequence[int], cluster_b: Sequence[int]):
        """
        Computes the distance between two clusters as the minimum among all
        inter-cluster pair-wise distances.

        :param cluster_a: a cluster
        :param cluster_b: another cluster
        :return: the distance
        """

        min_distance = float("inf")
        for i in cluster_a:
            for j in cluster_b:
                d = distance.euclidean(self.data.iloc[i], self.data.iloc[j])
                if min_distance > d:
                    min_distance = d

        return min_distance

    def nn_cluster(self, cl_key: int, cluster_index: Sequence[int]):
        """Finds the nearest neighbor to a cluster
        :param cluster_index: List of indexes contained in the list
        :param cl_key: The key of the list in the cluster dictionary
        """
        global x
        dist = float("inf")
        candidate = []  # List that stores multiple candidates

        for key, value in self.clusters.items():  # For each other cluster

            if cl_key == key:  # Making sure its a different to our current cluster
                continue
            temp = self.__cluster_distance(cluster_index, value)  # Find their Distance

            if dist > temp:  # If its smaller than the previous, store it
                dist = temp
                candidate = [key]

            elif dist == temp:  # If its the same, store it as well
                candidate.append(key)
        x-=1
        return candidate

    def __rnn_cluster(self, e: int, cluster: List[int]):
        """
        Calculates the RNN of a cluster for a certain prefix.

        :param e: the prefix for which we want to find the RNN
        :param cluster: the cluster that we want to find the RNN
        """

        rnn = set()
        complete = set()
        for item in cluster:
            rnn.union(self.rnn[e][item])
        for item in rnn:
            if item not in cluster:
                complete.add(item)
        return complete

    def __mpl_calculation(self, cluster: List[int]):
        """Finds the MPL of discriminative clusters

        :param cluster: The cluster of which we want to find it's MPL"""
        # Checking if the support condition is met
        index = self.labels[cluster[0]]
        if self.support > len(cluster) / self.occur[index]:
            return
        mpl_rnn = self.timestamps[len(
            self.timestamps) - 1]  # Initializing the  variables that will indicate the minimum timestamp from which each rule applies
        mpl_nn = self.timestamps[len(self.timestamps) - 1]
        """Checking the RNN rule for the clusters"""

        curr_rnn = self.__rnn_cluster(self.timestamps[len(self.timestamps) - 1], cluster)  # Finding the RNN for the L

        if self.relaxed:
            curr_rnn = curr_rnn.intersection(self.correct)

        for e in reversed(self.timestamps):

            temp = self.__rnn_cluster(e, cluster)  # Finding the RNN for the next timestamp
            if self.relaxed:
                temp = temp.intersection(self.correct)

            if not curr_rnn - temp:  # If their division is an empty set, then the RNN is the same so the
                # MPL is e
                mpl_rnn = e
            else:
                break
            curr_rnn = temp

        """Then we check the 1-NN consistency"""
        rule_broken = 0
        for e in reversed(self.timestamps):  # For each timestamp

            for series in cluster:  # For each time-series

                for my_tuple in self.nn[e][series]:  # We check the corresponding NN to the series
                    if my_tuple not in cluster:
                        rule_broken = 1
                        break
                if rule_broken == 1:
                    break
            if rule_broken == 1:
                break
            else:
                mpl_nn = e
        for series in cluster:
            pos = max(mpl_rnn, mpl_nn)  # The value at which at least one rule is in effect
            if self.mpl[series] > pos:
                self.mpl[series] = pos

    def __mpl_clustering(self):
        """Executes the hierarchical clustering"""
        pool = Pool(mp.cpu_count())
        n = self.data.shape[0]
        redirect = {}  # References an old cluster pair candidate to its new place
        discriminative = 0  # Value that stores the number of discriminative values found
        """Initially make as many clusters as there are items"""
        for index, row in self.data.iterrows():
            self.clusters[index] = [index]
            redirect[index] = index
        result = []
        """Clustering loop"""
        while n > 1:  # For each item
            closest = {}
            my_list = list(self.clusters.items())
            res = pool.starmap(self.nn_cluster, my_list)
            for key,p  in zip(self.clusters.keys(),res):
                closest[key] = p
            logger.debug(closest)
            for key, value in closest.items():
                for item in list(value):
                    if key in closest[item]:  # Mutual pair found
                        closest[item].remove(key)
                        if  redirect[item]==redirect[key]:      #If 2 time-series are in the same cluster(in case they had an 3d  neighboor that invited them in the cluster)
                            continue
                        for time_series in self.clusters[redirect[item]]:
                            self.clusters[redirect[key]].append(time_series)  # Commence merging
                        del self.clusters[redirect[item]]
                        n = n - 1
                        redirect[item] = redirect[key]  # The item can now be found in another cluster
                        for element in self.clusters[redirect[key]]:  # Checking if cluster is discriminative
                            result.append(self.labels.loc[element])

                        x = np.array(result)
                        if len(np.unique(x)) == 1:  # If the unique class labels is 1, then the
                            # cluster is discriminative
                            discriminative += 1
                            self.__mpl_calculation(self.clusters[redirect[key]])

                        for neighboors_neigboor in closest:  # The items in the cluster that has been assimilated can
                            # be found in the super-cluster
                            if redirect[neighboors_neigboor] == item:
                                redirect[neighboors_neigboor] = key
                        result.clear()
            if discriminative == 0:  # No discriminative clusters found
                break
            discriminative = 0
        pool.terminate()
    def predict(self, test_data: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Prediction phase.
        Finds the 1-NN of the test data and if the MPL oof the closest time-series allows the prediction, then return that prediction
         """
        predictions = []
        nn = []
        candidates = []  # will hold the potential predictions
        cand_min_mpl = []
        #test_data = test_data.rename(columns=lambda x: x - 1)
        for test_index, test_row in test_data.iterrows():
            for e in self.timestamps:
                neigh = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(self.data.iloc[:, 0:e + 1])
                neighbors = neigh.kneighbors([test_row[0:e + 1]])
                candidates.clear()
                cand_min_mpl.clear()
                nn = neighbors[1]
                for i in nn:
                    if e >= self.mpl[i[0]]:
                        candidates.append((self.mpl[i[0]], self.labels[i[0]]))  # Storing candidates by mpl and by label
                if len(candidates) > 1:  # List is not empty so wee found candidates
                    candidates.sort(key=lambda x: x[0])
                    for candidate in candidates:

                        if candidate[0] == candidates[0][0]:
                            cand_min_mpl.append(candidate)  # Keeping the candidates with the minimum mpl
                        else:
                            break  # From here on the mpl is going to get bigger

                    predictions.append(
                        (e, max(set(cand_min_mpl), key=cand_min_mpl.count)))  # The second argument is the max label
                    break
                elif len(candidates) == 1:  # We don't need to to do the above if we have only one nn
                    predictions.append((e, candidates[0][1]))
                    break
            if candidates == 0:
                predictions.append((self.timestamps[-1], 0))
        return predictions

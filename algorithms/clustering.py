import pandas as pd
from sklearn.cluster import MiniBatchKMeans


class Clusterer:
    def __init__(self, data, n_clusters=8):
        self.data = data
        self.n_clusters = n_clusters

    def return_clusters(self):
        cluster_indexes = MiniBatchKMeans(
            n_clusters=self.n_clusters, batch_size=2048).fit_predict(pd.concat([self.data, self.get_fictive_variable(self.data)], axis=1))
        print('clusterization done')
        return self.get_clusters(cluster_indexes)

    def get_fictive_variable(self, data):
        dffv = pd.DataFrame(data['prime_profit'] /
                            data['pcc'], columns=['rentabilite'])
        dffv['b_sur_a'] = data['coeff_prix']/data['coeff_non_prix']
        return dffv

    def get_clusters(self, cluster_indexes):
        clusters = [pd.DataFrame()]*self.n_clusters
        data_copy = self.data.copy()
        for k in range(self.n_clusters):
            clusters[k] = data_copy[cluster_indexes == k]
        return clusters

    def reconstruct_data(self, clusters):
        new_data = pd.DataFrame()
        for data_cluster in clusters:
            new_data = new_data.append(data_cluster)
        new_data = new_data.sort_index(ascending=True)
        return new_data

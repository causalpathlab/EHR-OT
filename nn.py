import numpy as np
from sklearn.neighbors import NearestNeighbors


def NN_trans(source_reps, target_reps, n_neighbors, type):
    """ 
    Domain adaptation using nearest neighbor

    :param 2D np array(float) source_reps: source representations
    :param 2D np array(float) target_reps: target representations
    :param int n_neighbors: number of neighbors 
    :param str type: type is either classification or regression

    return the nearest source representation for target representations using source \
        representations and source labels via nearest neighbor. If the type is regression, \
        return one transported representation. If the type is classification, return the \
        number of neighbors representations
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(source_reps)
    nearest_neighbors_indices =  knn.kneighbors(target_reps)[1]
    trans_reps = source_reps[nearest_neighbors_indices]
    if type == 'regression':
        trans_reps = np.mean(trans_reps, axis=1)
    return trans_reps


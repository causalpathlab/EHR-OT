import numpy as np
from sklearn.decomposition import PCA


def coords2features(coords):
    """ 
    Convert coordinates to features
    :param 2d np array of floats coords: coordinates in the embedding space
    """

    features = []
    for coord in coords:
        features.append([round(x) for x in coord])
    return np.array(features)


def features2embs(source_features, target_features):
    """ 
    Convert features to embeddings using PCA without correcting the mean
    """

    source_pca = PCA(n_components=2)
    target_pca = PCA(n_components=2)
    source_embs = source_pca.fit_transform(source_features)
    source_embs = np.add(source_embs, [source_pca.mean_[0:2]]*source_embs.shape[0])
    target_embs = target_pca.fit_transform(target_features)
    target_embs = np.add(target_embs, [target_pca.mean_[0:2]]*target_embs.shape[0])


    return source_embs, target_embs


def special_div(x, y):
    """ 
    Special division operation
    """
    if y == 0:
        y = 1e-5
    return x/y
""" 
Implement Transfer Component Analysis
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer

# # Load source and target domain data
# X_source, y_source = load_source_data() # Load source domain data
# X_target, y_target = load_target_data() # Load target domain data


def ESDA(X_source, y_source, X_target, y_target, n_components, model_fn):
    # Standardize the data
    scaler = StandardScaler()
    X_source = scaler.fit_transform(X_source)
    X_target = scaler.fit_transform(X_target)

    # Perform PCA on source domain data
    pca = PCA(n_components=n_components)
    X_source_pca = pca.fit_transform(X_source)

    # Perform Euclidean space data alignment
    

    # Compute the kernel matrices
    K_source = X_source_pca @ X_source_pca.T
    K_target = X_target @ X_source_pca.T

    # Center the kernel matrices
    centerer = KernelCenterer()
    K_source_centered = centerer.fit_transform(K_source)
    K_target_centered = centerer.transform(K_target)

    # Perform eigendecomposition on the kernel matrices
    eigenvalues, eigenvectors = np.linalg.eig(K_source_centered @ K_target_centered.T)

    # Sort the eigenvectors based on eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute the transformation matrix for Euclidean space data alignment
    W = K_source_centered.T @ eigenvectors @ np.diag(1 / np.sqrt(eigenvalues))

    # Transform the source and target domain data to the aligned feature space
    X_source_aligned = X_source @ W
    X_target_aligned = X_target @ W

    # Now you can use X_source_train_aligned, X_source_test_aligned, X_target_train_aligned, X_target_test_aligned
    # as the aligned features for further analysis or modeling, such as classification or regression.

    # Train a classifier on the aligned features
    clf = model_fn()
    clf.fit(X_source_aligned, y_source)
    accuracy = clf.score(X_target_aligned, y_target)
    print("Accuracy on target domain test set:", accuracy)


def TCA(X_source, y_source, X_target, y_target, model_fn):
    # Standardize the data
    scaler = StandardScaler()
    X_source = scaler.fit_transform(X_source)
    X_target = scaler.fit_transform(X_target)

    # Compute the kernel matrices
    K_source = rbf_kernel(X_source, gamma=1)
    K_target = rbf_kernel(X_target, X_source, gamma=1)

    # Center the kernel matrices
    centerer = KernelCenterer()
    K_source_centered = centerer.fit_transform(K_source)
    K_target_centered = centerer.transform(K_target)

    # Perform eigendecomposition on the kernel matrices
    eigenvalues, eigenvectors = np.linalg.eig(K_source_centered @ K_target_centered.T)

    # Sort the eigenvectors based on eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute the TCA components
    TCA_components = K_source_centered.T @ eigenvectors @ np.diag(1 / np.sqrt(eigenvalues))

    # Use TCA components as features for classification or other tasks
    print(X_source.shape, TCA_components.shape)
    X_source_tca = X_source @ TCA_components
    X_target_tca = X_target @ TCA_components

    # Now you can use X_source_train_tca, X_source_test_tca, X_target_train_tca, X_target_test_tca
    # as the aligned features for further analysis or modeling, such as classification or regression.
    clf = model_fn()
    clf.fit(X_source_tca, y_source)
    accuracy = clf.score(X_target_tca, y_target)
    return accuracy


def TCA2(Xs, Xt, n_components=None, scale=True):
    """
    Transfer Component Analysis (TCA) implementation.

    Parameters:
    -----------
    Xs : array-like, shape (n_samples_source, n_features)
        Source domain data.
    Xt : array-like, shape (n_samples_target, n_features)
        Target domain data.
    n_components : int or None, optional (default=None)
        Number of components to keep. If None, keeps all components.
    scale : bool, optional (default=True)
        Whether to perform feature scaling before applying TCA.

    Returns:
    --------
    Xs_tca : array, shape (n_samples_source, n_components)
        Transformed source domain data after TCA.
    Xt_tca : array, shape (n_samples_target, n_components)
        Transformed target domain data after TCA.
    """

    if scale:
        # Standardize the data
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs)
        Xt = scaler.transform(Xt)

    # Perform PCA on both source and target domains
    pca = PCA(n_components=n_components)
    Xs_pca = pca.fit_transform(Xs)
    Xt_pca = pca.transform(Xt)

    # Center the PCA transformed source and target data
    Xs_mean = np.mean(Xs_pca, axis=0)
    Xt_mean = np.mean(Xt_pca, axis=0)
    Xs_pca_centered = Xs_pca - Xs_mean
    Xt_pca_centered = Xt_pca - Xt_mean

    # Compute the covariance matrices of centered PCA transformed source and target data
    cov_source = (1 / (Xs_pca_centered.shape[0] - 1)) * np.dot(Xs_pca_centered.T, Xs_pca_centered)
    cov_target = (1 / (Xt_pca_centered.shape[0] - 1)) * np.dot(Xt_pca_centered.T, Xt_pca_centered)

    # Perform singular value decomposition (SVD) on the covariance matrices
    U, _, Vt = np.linalg.svd(cov_source.T @ cov_target)

    # Compute the optimal projection matrix
    W = np.dot(U, Vt)

    # Transform the source and target data using the optimal projection matrix
    Xs_tca = np.dot(Xs_pca, W.T)
    Xt_tca = np.dot(Xt_pca, W.T)

    return Xs_tca, Xt_tca

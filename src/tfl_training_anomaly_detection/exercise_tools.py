import itertools as it
import logging
import os
from math import ceil

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import fetch_kddcup99, fetch_openml, make_spd_matrix
from sklearn.ensemble import IsolationForest as iForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from libs import data, kde_lib
from libs.exp_lib import Density_model

tfd = tfp.distributions

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from plotly import express as px

logger = logging.getLogger(__name__)


def check_is_testing_pipeline():
    return os.getenv("IS_TEST_CI_PIPELINE", "False") in [
        "True",
        "TRUE",
        "T",
        "yes",
        "YES",
        "Y",
        "1",
    ]


# Helper function
def visualize_kde(kernel: str, bandwidth: float, X_train: np.array, y_train: np.array):
    """
    Visualize kde

    :param kernel: the kde kernel
    :param bandwidth: the kde bandwidth
    :param X_train: training data
    :param y_train: training labels (0=nominal, 1=anomaly)
    """
    fig, axis = plt.subplots(figsize=(5, 5))
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kde.fit(X_train)

    lin = np.linspace(-10, 10, 50)
    grid_points = list(it.product(lin, lin))
    ys, xs = np.meshgrid(lin, lin)
    # The score function of sklearn returns log-densities
    scores = np.exp(kde.score_samples(grid_points)).reshape(50, 50)
    colormesh = axis.contourf(xs, ys, scores)
    fig.colorbar(colormesh)
    axis.set_title("Density Contours (Bandwidth={})".format(bandwidth))
    axis.set_aspect("equal")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.show()


def visualize_mahalanobis(data, y: pd.Series, scores, mu, sigma_diag, thr):
    """
    Visualizes the Mahalanobis distance

    :param data: the data set
    :param y: labels (0=nominal, 1=anomaly)
    :param mu: mean for the Mahalanobis distance
    :param sigma_diag: diagonal vector of the covariance matrix
    :param thr: the threshold to classify a point as anomaly
    """
    _, axes = plt.subplots(figsize=(6, 6))

    # Visualize Data
    scatter_gt = axes.scatter(data[:, 0], data[:, 1], c=y)
    plt.scatter(mu[0], mu[1], color="red")
    axes.set_title("Ground Truth")
    handles, _ = scatter_gt.legend_elements()
    axes.legend(handles, ["Nominal", "Anomaly"])
    axes.set_aspect("equal")
    # Draw decision contour
    decision_boundary = Ellipse(
        mu,
        width=2 * np.sqrt(sigma_diag[0]) * thr,
        height=2 * np.sqrt(sigma_diag[1]) * thr,
        color="red",
        fill=False,
    )
    axes.add_patch(decision_boundary)

    # Evaluate threshold
    y_pred = scores > thr

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    axes.set_title("Precision: {}\nRecall: {}\nF1: {}".format(precision, recall, f1))

    plt.tight_layout()
    plt.show()


def get_kdd_data():
    """
    Download KDD dataset. Provides labels only for the test set.

    :return: X_train, X_test, y_test
    """
    is_testing = check_is_testing_pipeline()
    if is_testing:
        logger.info("Loading reduced kdd cup data for testing pipeline.")
        import pickle as pkl

        with open("../../data/kddcup99/kddcup99_trial.pkl", "rb") as f:
            KDD99_trial = pkl.load(f)
        X = KDD99_trial["data"]
        y = KDD99_trial["target"]
    else:
        KDD99 = fetch_kddcup99(subset="SA")

        X = KDD99["data"]
        y = KDD99["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return [X_train, X_test, y_test]


def evaluate(y_true, y_pred, axes=None, save_as=None):
    """
    Compute and display the ROC and PR curve as well as ROC AUC and AP

    :param y_true: Ground truth
    :param y_pred: Predictions
    :param save_as: Location for the figure to be saved. If save_as is None then the figure is not saved
    :return: None
    """
    if axes is None:
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    else:
        for ax in axes:
            ax.clear()

    # ROC Curve
    fp, tp, roc_thr = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    axes[0].plot(fp, tp)
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_title("ROC Curve\n ROC AUC: {:.2}".format(auc))

    # PR Curve
    prec, rec, pr_thr = precision_recall_curve(y_true, y_pred)
    auc = average_precision_score(y_true, y_pred)
    axes[1].plot(rec, prec)
    axes[1].set_ylabel("Precision")
    axes[1].set_xlabel("Recall")
    axes[1].set_title("PR Curve\n AP: {:.2}".format(auc))

    if save_as is not None:
        plt.savefig(save_as)
    plt.show()
    return {"ROC": [fp, tp, roc_thr], "PR": [prec, rec, pr_thr]}


def create_distributions(dim=2, dim_irrelevant=0):
    """
    Create the base distributions for the exercise

    :param dim: Base Dimensionality
    :param dim_irrelevant: Additional noise dimensions
    :return: Dictionary of distributions
    """
    distributions = dict()

    #
    # Uniform
    #
    low = [-10] * dim
    high = [10] * dim
    uniform = tfd.Independent(tfd.Uniform(low=low, high=high))

    if dim_irrelevant > 0:
        uniform = tfd.JointDistributionSequential(
            [
                uniform,
                tfd.MultivariateNormalDiag(
                    loc=[0.0] * dim_irrelevant, scale_diag=[0.1] * dim_irrelevant
                ),
                lambda i, u: tfd.Deterministic(tf.concat([u, i], axis=1)),
            ]
        )
    else:
        uniform = tfd.JointDistributionSequential([uniform])

    distributions["Uniform"] = uniform

    #
    # Single Gaussian
    #
    mean_normal = ([0.0] * (dim - 1)) + [-2.0]
    covariance_normal = tf.linalg.cholesky(np.identity(dim) * 0.5)
    if dim_irrelevant > 0:
        normal = tfd.JointDistributionSequential(
            [
                tfd.MultivariateNormalTriL(mean_normal, covariance_normal),
                tfd.MultivariateNormalDiag(
                    loc=[0.0] * dim_irrelevant, scale_diag=[0.1] * dim_irrelevant
                ),
                lambda i, u: tfd.Deterministic(
                    tf.concat([u, tf.cast(i, tf.double)], axis=1)
                ),
            ]
        )
    else:
        normal = tfd.JointDistributionSequential(
            [tfd.MultivariateNormalTriL(mean_normal, covariance_normal)]
        )

    distributions["Blob"] = normal

    #
    # Mixture of Two Gaussians
    #
    probs_gaussians = [0.5, 0.5]
    mean_gaussians = [[5] * dim, [-5] * dim]
    covariance_gaussians = [
        tf.linalg.cholesky(np.identity(dim)),
        tf.linalg.cholesky(np.identity(dim)),
    ]
    gaussian_mixture = tfd.MixtureSameFamily(
        tfd.Categorical(probs=probs_gaussians),
        tfd.MultivariateNormalTriL(mean_gaussians, covariance_gaussians),
    )

    if dim_irrelevant > 0:
        gaussian_mixture = tfd.JointDistributionSequential(
            [
                gaussian_mixture,
                tfd.MultivariateNormalDiag(
                    loc=[0.0] * dim_irrelevant, scale_diag=[0.1] * dim_irrelevant
                ),
                lambda i, u: tfd.Deterministic(
                    tf.concat([u, tf.cast(i, tf.double)], axis=1)
                ),
            ]
        )
    else:
        gaussian_mixture = tfd.JointDistributionSequential([gaussian_mixture])

    distributions["Double Blob"] = gaussian_mixture

    #
    # Circular shape
    #
    radius = 6
    thickness_circle = 0.5

    if dim_irrelevant > 0:
        circle = tfd.JointDistributionSequential(
            [
                tfd.MultivariateNormalDiag(loc=[0.0] * dim, scale_diag=[1.0] * dim),
                tfd.MultivariateNormalDiag(
                    loc=[0.0] * dim, scale_diag=[thickness_circle] * dim
                ),
                lambda p, a: tfd.Deterministic(
                    (radius * a / tf.reshape(np.linalg.norm(a, axis=1), (-1, 1))) + p
                ),
                tfd.MultivariateNormalDiag(
                    loc=[0.0] * dim_irrelevant, scale_diag=[0.1] * dim_irrelevant
                ),
                lambda i, u: tfd.Deterministic(tf.concat([u, i], axis=1)),
            ]
        )
    else:
        circle = tfd.JointDistributionSequential(
            [
                tfd.MultivariateNormalDiag(loc=[0.0] * dim, scale_diag=[1.0] * dim),
                tfd.MultivariateNormalDiag(
                    loc=[0.0] * dim, scale_diag=[thickness_circle] * dim
                ),
                lambda p, a: tfd.Deterministic(
                    (radius * a / tf.reshape(np.linalg.norm(a, axis=1), (-1, 1))) + p
                ),
            ]
        )
    distributions["Sphere"] = circle

    #
    # Sinusoidal shape
    #
    low = -2 * np.pi
    high = 2 * np.pi
    thickness_sinusoidal = 0.5

    def create_point(x, epsilon):
        """
        Helper function that creates points on the sinusoidal shape

        :param x: point description
        :param epsilon: noise to be added
        """
        res = np.array(
            tf.concat(
                [
                    tf.reshape(x + epsilon[:, :-1], (len(x), -1)),
                    tf.reshape(
                        2 * np.cos(np.linalg.norm(x, axis=1)) + epsilon[:, -1],
                        (len(x), -1),
                    ),
                ],
                axis=1,
            )
        )
        return res

    if dim_irrelevant > 0:
        sinusoidal = tfd.JointDistributionSequential(
            [
                tfd.Independent(
                    tfd.Uniform(low=[low] * (dim - 1), high=[high] * (dim - 1))
                ),
                tfd.MultivariateNormalDiag(
                    loc=[0.0] * dim, scale_diag=[thickness_sinusoidal] * dim
                ),
                lambda n, u: tfd.Deterministic(create_point(u, n)),
                tfd.MultivariateNormalDiag(
                    loc=[0.0] * dim_irrelevant, scale_diag=[0.1] * dim_irrelevant
                ),
                lambda i, u: tfd.Deterministic(tf.concat([u, i], axis=1)),
            ]
        )

    else:
        sinusoidal = tfd.JointDistributionSequential(
            [
                tfd.Independent(
                    tfd.Uniform(low=[low] * (dim - 1), high=[high] * (dim - 1))
                ),
                tfd.MultivariateNormalDiag(
                    loc=[0.0] * dim, scale_diag=[thickness_sinusoidal] * dim
                ),
                lambda n, u: tfd.Deterministic(create_point(u, n)),
            ]
        )
    distributions["Sinusoidal"] = sinusoidal

    #
    # Random Gaussian Mixture
    #
    n_components = 5
    probs = [1 / n_components] * n_components
    means = uniform = tfd.Independent(
        tfd.Uniform(low=[-8] * dim, high=[8] * dim)
    ).sample(n_components)
    covariances = [
        tf.linalg.cholesky(make_spd_matrix(dim).astype(np.float32))
        for _ in range(n_components)
    ]

    random_mixture = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=probs),
        components_distribution=tfd.MultivariateNormalTriL(
            loc=means, scale_tril=covariances
        ),
    )

    if dim_irrelevant > 0:
        random_mixture = tfd.JointDistributionSequential(
            [
                random_mixture,
                tfd.MultivariateNormalDiag(
                    loc=[0.0] * dim_irrelevant, scale_diag=[0.1] * dim_irrelevant
                ),
                lambda i, u: tfd.Deterministic(tf.concat([u, i], axis=1)),
            ]
        )
    else:
        random_mixture = tfd.JointDistributionSequential([random_mixture])

    distributions["Random Gaussian Mixture"] = random_mixture

    return distributions


def contamination(nominal, anomaly, p):
    """
    Build the mixture model (1-@p)*@nominal + @p*@anomaly

    :param nominal: Nominal Distribution
    :param anomaly: Anomaly Distribution
    :param p: Probability of anomaly
    :return: Mixture model
    """
    # Build explicit mixture model
    return tfd.JointDistributionSequential(
        [
            tfd.Bernoulli(probs=p, dtype=tf.double),
            nominal,
            anomaly,
            lambda a, n, b: tfd.Deterministic(
                (tf.reshape(b, (-1, 1)) * tf.cast(a[-1], tf.double))
                + (tf.reshape((1 - b), (-1, 1)) * tf.cast(n[-1], tf.double))
            ),
        ]
    )


def get_house_prices_data(neighborhood="CollgCr", anomaly_neighborhood="NoRidge"):
    """
    Load house prices data for one neighborhood.
    The method returns also test data which are made of data from selected neighborhood plus data
    from another neighborhood, considered as anomaly.

    :param neighborhood: str, key corresponding to neighborhood.
    :param anomaly_neighborhood: neighborhood to use as anomaly. Must be different to neighborhood
    :return: train data, i.e. data only from selected neighborhood, test data, i.e. data with
        contamination, and test labels, i.e. a list of zeros and one corresponding to normal or
        anomalous data respectively.
    """
    house_data = pd.read_csv("../data/house_prices/house_prices.csv")
    neighborhood_data = house_data[house_data["Neighborhood"] == neighborhood].drop(
        columns=["Neighborhood"]
    )
    X_train, X_test = train_test_split(neighborhood_data, test_size=0.2)
    X_anomalies = house_data[house_data["Neighborhood"] == anomaly_neighborhood].drop(
        columns=["Neighborhood"]
    )
    y_test = [0] * len(X_test) + [1] * len(X_anomalies)
    X_test = X_test.append(X_anomalies, ignore_index=True)
    X_test, y_test = shuffle(X_test, y_test)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_test


def get_mnist_data():
    """
    Loads mnist data into pandas Dataframe with columns data and target.
    If dataset is not already cached, fetching it through openml could give some problems with the SSL Certificate.
    So, if SSLCertVerificationError is raised, run the following:

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    get_mnist_data()
    """
    is_testing = check_is_testing_pipeline()
    if is_testing:
        logger.info("Loading reduced mnist data for testing pipeline.")
        raw_mnist = pd.read_csv("../../data/mnist/mnist_784_trial.csv")
        target = raw_mnist.values[:, -1]
        data = raw_mnist.values[:, :-1]
    else:
        raw_mnist = fetch_openml("mnist_784", version=1)
        data = raw_mnist.data.values
        target = raw_mnist.target
    mnist = {"data": data, "target": target}
    return mnist


def benchmark_algorithms():
    """
    Conduct a series of experiments with KDE and iForest and return result summary
    Experiment overview:
    - Nominal Distributions: Double Blob, Sphere, Sinusoidal, and Random Gaussian Mixture
    - Anomaly Disdributions: Uniform and Blob
    - Number of Base Dimensions: 2 - 20
    - Irrelevant Dimensions: Gaussian Noise (variance = 0.1)
    - Number of Irrelevant Dimansions: 0 - 10
    - Contamination level: 1% - 20%

    :return: DataFrame with ROC AUC and AP for each experiment
    """
    n_samples = 1000
    dimensions = [2, 5, 20]
    irrelevant_dimensions = [0, 10]
    nominals = ["Double Blob", "Sinusoidal", "Sphere", "Random Gaussian Mixture"]
    anomalies = ["Blob", "Uniform"]
    ps = [0.01, 0.05, 0.1, 0.2]
    runs = 5
    result = []
    for dim in dimensions:
        for idim in irrelevant_dimensions:
            dists = create_distributions(dim, idim)
            for nom in nominals:
                for ano in anomalies:
                    for p in ps:
                        auc_if = []
                        ap_if = []
                        auc_kde = []
                        ap_kde = []

                        dist = contamination(dists[nom], dists[ano], p)
                        for _ in range(runs):
                            sample = dist.sample(n_samples)
                            X = sample[-1]
                            y = sample[0]

                            sample = dist.sample(n_samples)
                            X_test = sample[-1]
                            y_test = sample[0]

                            # iForest
                            iforest = iForest()
                            iforest.fit(np.array(X))
                            scores = iforest.decision_function(np.array(X_test))
                            auc_if.append(roc_auc_score(y_test, scores))
                            ap_if.append(average_precision_score(y_test, scores))

                            # KDE
                            kde = KernelDensity(
                                kernel="gaussian",
                                bandwidth=(dim + idim) / np.sqrt(n_samples),
                            )
                            kde.fit(X)
                            scores = 1 - np.exp(kde.score_samples(X_test))
                            auc_kde.append(roc_auc_score(y_test, scores))
                            ap_kde.append(average_precision_score(y_test, scores))

                        result.append(
                            [
                                "iForest",
                                dim + idim,
                                idim,
                                nom,
                                ano,
                                p,
                                np.average(auc_if),
                                np.std(auc_if),
                                np.average(ap_if),
                                np.std(ap_if),
                            ]
                        )
                        result.append(
                            [
                                "KDE",
                                dim + idim,
                                idim,
                                nom,
                                ano,
                                p,
                                np.average(auc_kde),
                                np.std(auc_kde),
                                np.average(ap_kde),
                                np.std(ap_kde),
                            ]
                        )

    result = pd.DataFrame(
        np.array(result),
        columns=[
            "Algorithm",
            "Dimensions",
            "Irrelevant Dimensions",
            "Nominal",
            "Anomaly",
            "p",
            "ROC AUC",
            "STD ROC AUC",
            "AP",
            "STD AP",
        ],
    )
    result.to_csv("benchmark.csv")
    return result


def anomaly_from_classification(data, target, nominal_classes, anomaly_classes, p):
    """
    Build an anomaly detection data set from a classification data set. The new data set contains
    only binary labels where 0 denotes the nominal class.

    :param data: Data
    :param target: Target
    :param nominal_classes: List like, the nominal classes
    :param anomaly_classes: List like, the anomaly classes
    :param p: Fraction of anomalies
    :return: X, y; Subset X of data randomly chosen according to input parameters and binary target vector y (0=nominal)
    """

    df = pd.DataFrame(data)
    df["Target"] = target

    result = df[df.apply(lambda x: x["Target"] in nominal_classes, axis=1)]

    # n_anomalies/(n_nominals + n_anomalies) = p
    n_nominals = len(result)
    n_anomalies = (p / (1 - p)) * n_nominals
    n_anomalies = ceil(n_anomalies)
    anomalies = df[df.apply(lambda x: x["Target"] in anomaly_classes, axis=1)].sample(
        n_anomalies
    )

    result = result.append(anomalies)
    result = result.sample(frac=1)

    y = result.apply(
        lambda x: 0 if x["Target"] in nominal_classes else 1, axis=1
    ).values
    X = result.drop("Target", axis=1).values

    return [X, y]


def load_contaminated_data(dataset, dataset_options):
    if dataset == "house-prices":
        X_train, X_test, y_test = get_house_prices_data(**dataset_options)
        X = X_train.append(X_test, ignore_index=True)
        y = np.array([0] * len(X_train) + y_test)
        X, y = shuffle(X, y)
        # for the rkde experiment, 0 must be the label for anomalies, and 1 for normal data
        y = np.abs(y - 1)
    else:
        X, y = data.load_data_outlier(dataset)
    return X, y


######################################################################
# The following code is based on https://github.com/lminvielle/mom-kde
# see libs/licence for further information
######################################################################


def perform_rkde_experiment(
    algos,
    dataset,
    dataset_options,
    outlierprop_range,
    kernel,
):
    """
    Replicates the experiments from https://github.com/lminvielle/mom-kde
    """
    # =======================================================
    #   Processing
    # =======================================================

    print("Dataset: ", dataset)
    X0, y0 = load_contaminated_data(dataset, dataset_options)
    total_scores = pd.DataFrame()

    # Plot data is possiblr
    if dataset == "banana":
        plt.scatter(X0[:, 0], X0[:, 1], c=y0)
        plt.show()
    if dataset == "titanic":
        fig = px.scatter_3d(x=X0[:, 0], y=X0[:, 1], z=X0[:, 2], color=y0)
        fig.show()
    if dataset == "house-prices":
        fig = px.scatter_3d(
            X0, x="LotArea", y="OverallCond", z="SalePrice", color=np.abs(y0 - 1)
        )
        fig.show()

    scaler = MinMaxScaler()
    X0 = scaler.fit_transform(X0)
    for i_outlierprop, outlier_prop in enumerate(outlierprop_range):
        epsilon = outlier_prop / (1 - outlier_prop)
        print(
            "\nOutlier prop: {} ({} / {})".format(
                outlier_prop, i_outlierprop + 1, len(outlierprop_range)
            )
        )
        X, y = data.balance_outlier(X0, y0, e=epsilon)
        n_outliers = np.sum(y == 0)
        if epsilon == 0:
            k_range = [1]
        else:
            if epsilon < 1 / 3:
                k_max = 2 * n_outliers + 1
            else:
                k_max = X.shape[0] / 2
            k_range = np.linspace(1, k_max, 5).astype(int)

        # Find bandwidth
        h_cv, _, _ = kde_lib.bandwidth_cvgrid(X, kernel=kernel)

        # Processing all algos
        for algo in algos:
            print("Algo: ", algo)
            model = Density_model(algo, dataset, outlier_prop, kernel, h_cv)
            # if mom, run on several k
            if algo == "mom-kde":
                k_range_run = k_range
            else:
                k_range_run = [1]
            for k in k_range_run:
                model.fit(X, X, grid=None, k=k, norm_mom=False)
                if epsilon != 0:
                    model.compute_anomaly_roc(y)
                new_scores = model.get_score()
                total_scores = total_scores.append(new_scores, ignore_index=True)
    return total_scores


# =======================================================
#   Useful functions
# =======================================================


def set_datasetname(dataset):
    """
    Set dataset name
    """
    return dataset.replace("_", "-")


# Set the metric
metricname = {
    "auc_anomaly": "AUC",
    "jensen": "$D_{{JS}}$",
    "kullback_f0_f": "$D_{{KL}}$",
    "kullback_f_f0": "$D_{{KL}}$",
}

# Set the algorithm name
algoname = {"mom-kde": "MoM-KDE", "kde": "KDE", "spkde": "SPKDE", "rkde": "RKDE"}

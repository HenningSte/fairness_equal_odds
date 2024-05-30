from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.datasets import AdultDataset, GermanDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_compas,
)
from aif360.datasets import StructuredDataset, BinaryLabelDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from tqdm import tqdm


def aif_dataset_loader(dataset_used, protected_attribute_used):
    """
    Load the specified dataset from aif360 and split it into training, validation, and testing sets
    """
    dataset = None
    if dataset_used == "adult":
        dataset = AdultDataset()
        if protected_attribute_used == 1:
            privileged_groups = [{"sex": 1}]
            unprivileged_groups = [{"sex": 0}]
        else:
            privileged_groups = [{"race": 1}]
            unprivileged_groups = [{"race": 0}]

    elif dataset_used == "german":
        dataset = GermanDataset()
        if protected_attribute_used == 1:
            privileged_groups = [{"sex": 1}]
            unprivileged_groups = [{"sex": 0}]
        else:
            privileged_groups = [{"age": 1}]
            unprivileged_groups = [{"age": 0}]

    elif dataset_used == "compas":
        dataset = load_preproc_data_compas()
        if protected_attribute_used == 1:
            privileged_groups = [{"sex": 1}]
            unprivileged_groups = [{"sex": 0}]
        else:
            privileged_groups = [{"race": 1}]
            unprivileged_groups = [{"race": 0}]

    # Spelling error handling
    if dataset is None:
        raise ValueError("dataset must be one of 'adult', 'german', or 'compas'")

    # split dataset
    dataset_train, dataset_vt = dataset.split([0.6], shuffle=True)
    dataset_valid, dataset_test = dataset_vt.split([0.5], shuffle=True)

    return (
        dataset_train,
        dataset_valid,
        dataset_test,
        privileged_groups,
        unprivileged_groups,
    )


def classify(dataset_train, dataset_valid, dataset_test):
    """
    Regression classifier from aif360 demo
    https://github.com/Trusted-AI/AIF360/blob/main/examples/demo_calibrated_eqodds_postprocessing.ipynb
    """
    # Placeholder for predicted and transformed datasets
    dataset_train_pred = dataset_train.copy(deepcopy=True)
    dataset_valid_pred = dataset_valid.copy(deepcopy=True)
    dataset_test_pred = dataset_test.copy(deepcopy=True)

    # Logistic regression classifier and predictions for training data
    scale = StandardScaler()
    X_train = scale.fit_transform(dataset_train.features)
    y_train = dataset_train.labels.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)

    fav_idx = np.where(lmod.classes_ == dataset_train.favorable_label)[0][0]
    y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]

    # Prediction probs for validation and testing data
    X_valid = scale.transform(dataset_valid.features)
    y_valid_pred_prob = lmod.predict_proba(X_valid)[:, fav_idx]

    X_test = scale.transform(dataset_test.features)
    y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]

    class_thresh = 0.5
    dataset_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
    dataset_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)
    dataset_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

    y_train_pred = np.zeros_like(dataset_train_pred.labels)
    y_train_pred[y_train_pred_prob >= class_thresh] = dataset_train_pred.favorable_label
    y_train_pred[~(y_train_pred_prob >= class_thresh)] = (
        dataset_train_pred.unfavorable_label
    )
    dataset_train_pred.labels = y_train_pred

    y_valid_pred = np.zeros_like(dataset_valid_pred.labels)
    y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_valid_pred.favorable_label
    y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = (
        dataset_valid_pred.unfavorable_label
    )
    dataset_valid_pred.labels = y_valid_pred

    y_test_pred = np.zeros_like(dataset_test_pred.labels)
    y_test_pred[y_test_pred_prob >= class_thresh] = dataset_test_pred.favorable_label
    y_test_pred[~(y_test_pred_prob >= class_thresh)] = (
        dataset_test_pred.unfavorable_label
    )
    dataset_test_pred.labels = y_test_pred

    return (
        dataset_train_pred,
        dataset_valid_pred,
        dataset_test_pred,
    )


def prediction(weights, model, input_data):
    """
    Get model prediction
    """
    return model.apply(weights, input_data)


def get_performance_metrics(
    dataset, predictions, unprivileged_groups, privileged_groups
):
    """
    Get the accuracy and fairness of the given predictions
    """
    cm = ClassificationMetric(
        dataset,
        predictions,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    accuracy = cm.accuracy()
    fairness = cm.equal_opportunity_difference()
    return accuracy, fairness


def get_model_prediction_dataframes(data, weights, model):
    """
    Get the prediction dataframe from the preprocessed tensorflow dataset + the model predictions
    """
    # Load data as a pandas dataframe from tensorflow dataset
    data = np.array([sample[0].numpy() for sample in data.unbatch()])
    pred_df = pd.DataFrame(
        data=data,
        columns=[
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ],
    )

    # Add prediction column from model predictions
    pred_df = pred_df.assign(
        quality=[prediction(weights, model, data[i]) for i in tqdm(range(len(data)))]
    )
    pred_df["quality"] = pred_df["quality"].explode()
    pred_df["quality"] = pred_df["quality"].astype("float32")

    return pred_df


def get_ground_truth_dataframes(data):
    """
    Get the ground truth dataframe from the preprocessed tensorflow dataset
    """
    # Load dataset as a pandas dataframe
    df = tfds.as_dataframe(data.unbatch(), tfds.builder("wine_quality/white").info)

    # Reshape dataframe from features and labels
    cols = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]

    new_df = pd.DataFrame(df["features"].tolist(), index=df.index, columns=cols)
    df[cols] = new_df
    df.drop(columns=["features"], inplace=True)
    df["quality"] = df["quality"].astype("float32")

    return df


def create_aif360_dataset(df):
    """
    Create a binary label dataset from a pandas dataframe
    """
    # Create a binary label
    label_avg = np.average(df["quality"])
    df["quality"].where(df["quality"] >= label_avg, 0, inplace=True)
    df["quality"].where(df["quality"] < label_avg, 1, inplace=True)

    # Create a binary protected attribute
    protected_attr_avg = np.average(df["residual sugar"])
    df["residual sugar"].where(
        df["residual sugar"] >= protected_attr_avg, 0, inplace=True
    )
    df["residual sugar"].where(
        df["residual sugar"] < protected_attr_avg, 1, inplace=True
    )

    # Create a Structured Dataset
    dataset = StructuredDataset(
        df, label_names=["quality"], protected_attribute_names=["residual sugar"]
    )
    # Create a Binary Label Dataset
    dataset = BinaryLabelDataset(
        df=dataset.convert_to_dataframe()[0],
        label_names=dataset.label_names,
        protected_attribute_names=dataset.protected_attribute_names,
        favorable_label=1,
        unfavorable_label=0,
    )

    return dataset


def set_labels_from_scores(
    dataset,
    old_predictions,
    thresh,
    avg_odds,
    bal_acc,
    unprivileged_groups,
    privileged_groups,
):
    """
    Compute the labels based on the threshold value and probability score of the datapoint and add to metrics list
    """
    predictions = old_predictions.copy(deepcopy=True)

    # Blabla
    y_temp = np.zeros_like(predictions.labels)
    y_temp[predictions.scores >= thresh] = predictions.favorable_label
    y_temp[~(predictions.scores >= thresh)] = predictions.unfavorable_label
    predictions.labels = y_temp

    # Metrics for original validation data
    cm = ClassificationMetric(
        dataset,
        predictions,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    avg_odds.append(cm.equal_opportunity_difference())

    bal_acc.append(0.5 * (cm.true_positive_rate() + cm.true_negative_rate()))

    return avg_odds, bal_acc


def compute_ceop_output(
    all_thresh,
    dataset_orig_valid,
    dataset_orig_test,
    dataset_orig_valid_pred,
    dataset_orig_test_pred,
    dataset_transf_valid_pred,
    dataset_transf_test_pred,
    unprivileged_groups,
    privileged_groups,
):
    """
    Compute the calibrated equalized odds models prediction according to the given thresholds
    """
    bef_avg_odds_diff_valid = []
    bef_avg_odds_diff_test = []
    aft_avg_odds_diff_valid = []
    aft_avg_odds_diff_test = []
    bef_bal_acc_valid = []
    bef_bal_acc_test = []
    aft_bal_acc_valid = []
    aft_bal_acc_test = []

    for thresh in tqdm(all_thresh):
        bef_avg_odds_diff_valid, bef_bal_acc_valid = set_labels_from_scores(
            dataset_orig_valid,
            dataset_orig_valid_pred,
            thresh,
            bef_avg_odds_diff_valid,
            bef_bal_acc_valid,
            unprivileged_groups,
            privileged_groups,
        )
        bef_avg_odds_diff_test, bef_bal_acc_test = set_labels_from_scores(
            dataset_orig_test,
            dataset_orig_test_pred,
            thresh,
            bef_avg_odds_diff_test,
            bef_bal_acc_test,
            unprivileged_groups,
            privileged_groups,
        )
        aft_avg_odds_diff_valid, aft_bal_acc_valid = set_labels_from_scores(
            dataset_orig_valid,
            dataset_transf_valid_pred,
            thresh,
            aft_avg_odds_diff_valid,
            aft_bal_acc_valid,
            unprivileged_groups,
            privileged_groups,
        )
        aft_avg_odds_diff_test, aft_bal_acc_test = set_labels_from_scores(
            dataset_orig_test,
            dataset_transf_test_pred,
            thresh,
            aft_avg_odds_diff_test,
            aft_bal_acc_test,
            unprivileged_groups,
            privileged_groups,
        )
    return (
        bef_avg_odds_diff_valid,
        bef_avg_odds_diff_test,
        aft_avg_odds_diff_valid,
        aft_avg_odds_diff_test,
        bef_bal_acc_valid,
        bef_bal_acc_test,
        aft_bal_acc_valid,
        aft_bal_acc_test,
    )


def EOP_plotter(eop_handler):
    """
    Plot the eop results as 4 bar plots in plt
    """
    # Plot the eop results as 4 bar plots in plt
    # Get the data
    eop_original_acc_valid, eop_original_avg_odds_valid = np.absolute(
        eop_handler.get_dataset("validation")["original_performance_metrics"]
    )
    eop_post_acc_valid, eop_post_avg_odds_valid = np.absolute(
        eop_handler.get_dataset("validation")["transformed_performance_metrics"]
    )

    # Define the data
    accuracy_data = [eop_original_acc_valid, eop_post_acc_valid]
    odds_data = [eop_original_avg_odds_valid, eop_post_avg_odds_valid]
    labels = ["Original", "Transformed"]

    # Create a figure and two subplots
    fig = plt.figure(figsize=(8, 6))
    grid = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax1 = fig.add_subplot(grid[0, :])
    ax2 = fig.add_subplot(grid[1, :])

    # Plot the accuracy data
    x = np.arange(len(accuracy_data))
    ax1.bar(x, accuracy_data)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Comparison")

    accuracy_min = min(accuracy_data) - 0.01
    accuracy_max = max(accuracy_data) + 0.01
    ax1.set_ylim(accuracy_min, accuracy_max)

    # Plot the odds data
    x = np.arange(len(odds_data))
    ax2.bar(x, odds_data, color="r")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Equal opportunity diff")
    ax2.set_title("Equal opportunity difference Comparison")

    # Adjust the layout to prevent label clipping
    plt.tight_layout()

    # Display the plot
    plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_metrics(csv_file_path: str, output_file_path: str) -> List:
    """
    Read a CSV file with predicted and true labels, create the confusion matrix,
    visualize it, and save it to the specified output file.

    Args:
        csv_file_path (str): The file path of the CSV file containing predicted and true labels.
        output_file_path (str): The file path where the confusion matrix heatmap will be saved.

    Returns:
        list: A 2x2 confusion matrix represented as a list of lists.
            The rows and columns correspond to true positive, false negative,
            false positive, and true negative, respectively.
    """
    data = pd.read_csv(csv_file_path)

    predicted_labels = data["predicted_labels"]
    true_labels = data["true_labels"]

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    f1 = f1_score(true_labels, predicted_labels)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(output_file_path)
    plt.show()

    return conf_matrix, f1


def save_val_labels(
    val_predicted_labels: List[int], val_true_labels: List[int]
) -> None:
    """
    Save the validation predicted labels and true labels to a CSV file.

    Args:
        val_predicted_labels (List[int]): List of predicted labels for the validation set.
        val_true_labels (List[int]): List of true labels for the validation set.

    Returns:
        None
    """
    val_labels = np.hstack(
        (
            np.array(val_predicted_labels).reshape(-1, 1),
            np.array(val_true_labels).reshape(-1, 1),
        )
    )
    np.savetxt("val_labels.csv", val_labels, delimiter=",", fmt="%d")
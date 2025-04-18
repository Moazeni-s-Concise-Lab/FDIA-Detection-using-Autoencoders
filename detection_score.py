import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_reconstruction_error(true, pred):
    """
    Compute reconstruction error for each dataset splits (train, test and validation)

    Parameters:
    - true: np.ndarary or Tensor, true input data
    - pred: np.ndarray or Tensor, reconstructed input data
    
    Output:
    - reconstruction_error: np.ndarary or Tensor, average accross all input features with
    length equals to the length of the input data
    """
    deviations = (true - pred)**2
    reconstruction_error = deviations.sum(dim=1) / pred.shape[1]
    return reconstruction_error

def compute_attack_threshold(reconstruction_error, percentile=98.7):
    """
    Compute FDI attack detection threshold (tau) using the percentile that nicely
    detects all attacks in the validation data with least performance error  

    Parameters:
    - reconstruction_error: np.ndarary or Tensor, for validation dataset
    - percentile: a scaler, 98.7% is recommended
    
    Output:
    - tau: FDI attack detection threshold
    """
    tau = np.percentile(reconstruction_error, percentile)
    return tau

def get_predicted_labels(tau, reconstruction_error):
    """
    Compute predicted labels for each sample in the validation and test dataset
    if reconstruction error > tau, label the sample as attacked (1) otherwise 
    label it clean (0)

    Parameters:
    - reconstruction_error: Tensor, for validation and test dataset
    - tau: attach threshold
    
    Output:
    - pred_labels: predicted labels
    """
    
    pred_labels = (reconstruction_error > tau).int()
    return pred_labels

def detection_score(true_labels, pred_labels):
    """
    Compute confusion matrix for validation and test dataset and analyse 
    model performance.

    """
    cm = confusion_matrix(true_labels, pred_labels)
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]

    PPV = TP / (TP + FP)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    F1  = 2 * (PPV * TPR) / (PPV + TPR)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    BA  = (TPR + TNR) / 2

    return PPV, TPR, TNR, F1, Acc, BA

def plot_confusion_matrices(cm_test, cm_val):
    """
    Plots side-by-side confusion matrices for test and validation datasets.

    Parameters:
    - cm_test: np.ndarray or Tensor, confusion matrix for test data
    - cm_val: np.ndarray or Tensor, confusion matrix for validation data
    """
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(9, 3)
    fig.subplots_adjust(hspace=0.5, wspace=0.25)

    sns.heatmap(cm_test, annot=True, cmap='Greens', linewidths=0.5,
                linecolor='blue', fmt='g', annot_kws={'size': 10}, ax=ax[0])
    ax[0].set_title('Test Dataset', fontsize=10)
    ax[0].set_xlabel('Predicted Labels', fontsize=10)
    ax[0].set_ylabel('True Labels', fontsize=10)

    sns.heatmap(cm_val, annot=True, cmap='Greens', linewidths=0.5,
                linecolor='blue', fmt='g', annot_kws={'size': 10}, ax=ax[1])
    ax[1].set_title('Validation Dataset', fontsize=10)
    ax[1].set_xlabel('Predicted Labels', fontsize=10)
    ax[1].set_ylabel('True Labels', fontsize=10)

    plt.tight_layout()
    plt.show()
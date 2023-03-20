import numpy as np


def compute_confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the confusion matrix. The confusion
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    You do not need to implement confusion matrices for labels with more
    classes. You can assume this will always be a 2x2 matrix.

    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    true_negatives = int(0)
    false_positives = int(0)
    false_negatives = int(0)
    true_positives = int(0)
    size = int(predictions.shape[0])
    for i in range(size):
        if (bool(predictions[i]) == bool(actual[i]) and bool(predictions[i]) == True):
            true_positives = true_positives + 1
        elif (bool(predictions[i]) == bool(actual[i]) and bool(predictions[i]) == False):
            true_negatives = true_negatives + 1
        elif (bool(predictions[i]) != bool(actual[i]) and bool(predictions[i]) == True):
            false_positives = false_positives + 1
        elif (bool(predictions[i]) != bool(actual[i]) and bool(predictions[i]) == False):
            false_negatives = false_negatives + 1
    temp1 = [true_negatives, false_positives]
    temp2 = [false_negatives, true_positives]
    ans = [temp1, temp2]
    return ans

    raise NotImplementedError


def compute_accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the accuracy:

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    size = int(predictions.shape[0])
    confu_matrix = compute_confusion_matrix(actual=actual, predictions=predictions)
    accuracy = float((confu_matrix[0][0] + confu_matrix[1][1])/size)
    return accuracy

    raise NotImplementedError


def compute_precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    You MUST account for edge cases in which precision or recall are undefined
    by returning np.nan in place of the corresponding value.

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output a tuple containing:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    confu_matrix = compute_confusion_matrix(actual=actual, predictions=predictions)
    if (confu_matrix[0][1] + confu_matrix[1][1] == 0):
        precision = np.nan
        if (confu_matrix[0][1] + confu_matrix[1][0] == 0):
            recall = np.nan
        else:
            recall = confu_matrix[1][1]/(confu_matrix[1][1] + confu_matrix[1][0])
        tuple = (precision, recall)
    else:
        precision = confu_matrix[1][1]/(confu_matrix[0][1] + confu_matrix[1][1])
        if (confu_matrix[0][1] + confu_matrix[1][0] == 0):
            recall = np.nan
        else:
            recall = confu_matrix[1][1]/(confu_matrix[1][1] + confu_matrix[1][0])
        tuple = (precision, recall)
    return tuple

    raise NotImplementedError


def compute_f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the F1-measure:

    https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Because the F1-measure is computed from the precision and recall scores, you
    MUST handle undefined (NaN) precision or recall by returning np.nan. You
    should also consider the case in which precision and recall are both zero.

    Hint: implement and use the compute_precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    tuple = compute_precision_and_recall(actual=actual, predictions=predictions)
    precision = tuple[0]
    recall = tuple[1]
    if (np.isnan(precision) or np.isnan(recall)):
        return np.nan
    elif (precision == 0 and recall == 0):
        ans_t = float(0)
        return ans_t
    else:
        ans = float(2*precision*recall/(precision+recall))
        return ans
    raise NotImplementedError

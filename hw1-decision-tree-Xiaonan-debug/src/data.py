import numpy as np
import pandas as pd


def load_data(data_path):
    """
    This function loads the data in data_path csv into two numpy arrays:
    features (of size NxK) and targets (of size Nx1) where N is the number of rows
    and K is the number of features.

    Args:
        data_path (str): path to csv file containing the data

    Output:
        features (np.array): numpy array of size NxK containing the K features
        targets (np.array): numpy array of size 1xN containing the N targets.
        attribute_names (list): list of strings containing names of each attribute
            (headers of csv)
    """
    if data_path.endswith('gz'):
        df = pd.read_csv(data_path, compression='gzip')
    else:
        df = pd.read_csv(data_path)

    feature_columns = [col for col in df.columns if col != "class"]
    features = df[feature_columns].to_numpy()
    target = df[["class"]].to_numpy()
    a = 1

    return features, target, feature_columns


def train_test_split(features, targets, fraction):
    """
    Split features and targets into training and testing. The first M points
    from the data will be used for training and the remaining
    (features.shape[0] - M) points will be used for testing. Where M is:

        M = int(features.shape[0] * fraction)

    However, when fraction is 1.0, both training and test splits are
    the entire dataset. Code for this special case is provided for you.

    Args:
        features (np.array): NxD numpy array containing D features for each example
        targets (np.array): Nx1 numpy array containing labels corresponding to each example
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns (a tuple containing four variables):
        train_features: MxD numpy array of examples to be used for training
        train_targets: Mx1 numpy array of targets corresponding to `train_features`
        test_features: (N - M)xD numpy array of examples to be used for testing
        test_targets: (N - M)x1 numpy array of targets corresponding to `test_features`
    """

    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples!')
    elif fraction == 1.0:
        return features, targets, features, targets
    else:
        M = int(features.shape[0] * fraction)
        train_features = features[:M]
        train_targets = targets[:M]
        test_features = features[M:]
        test_targets = targets[M:]
        return train_features, train_targets, test_features, test_targets

    raise NotImplementedError


def cross_validation(features, targets, folds):
    """
    Split the data in `folds` different groups for cross-validation.
        Split the features and targets into a `folds` number of groups that
        divide the data as evenly as possible. Then for each group,
        return a tuple that treats that group as the test set and all
        other groups combine to make the training set.

        Note that this should be *deterministic*; don't shuffle the data.
        If there are 100 examples and you have 5 folds, each group
        should contain 20 examples and the first group should contain
        the first 20 examples.

        See test_cross_validation for expected behavior.

    Args:
        features: an NxK matrix of N examples, each with K features
        targets: an Nx1 array of N labels
        folds: the number of cross-validation groups

    Output:
        A list of tuples, where each tuple contains:
          (train_features, train_targets, test_features, test_targets)
    """

    assert features.shape[0] == targets.shape[0]
    if folds == 1:
        return [(features, targets, features, targets)]
    else:
        TList = []
        data_num = np.empty(int(folds), dtype=int)
        if(features.shape[0] % folds == 0):
            for i in range(folds):
                data_num[i] = features.shape[0] // folds
        else:
            con = folds - (features.shape[0] % folds)
            div = features.shape[0] // folds
            for i in range(folds):
                if(i >= con):
                    data_num[i] = div + 1
                else:
                    data_num[i] = div
        for i in range(folds):
            if(i == 0):
                test_features = features[:data_num[i]]
                test_targets = targets[:data_num[i]]
                train_features = features[data_num[i]:]
                train_targets = targets[data_num[i]:]
                tuple = (train_features, train_targets, test_features, test_targets)
                TList.append(tuple)
            elif(i > 0 and i < folds-1):
                test_features = features[data_num[i-1]:data_num[i-1] + data_num[i]]
                test_targets = targets[data_num[i-1]:data_num[i-1] + data_num[i]]
                train_features = np.append(features[:data_num[i-1]],features[data_num[i-1] + data_num[i]:], 0)
                train_targets = np.append(targets[:data_num[i-1]], targets[data_num[i-1] + data_num[i]:], 0)
                tuple = (train_features, train_targets, test_features, test_targets)
                TList.append(tuple)
                data_num[i] = data_num[i-1] + data_num[i]
            else:
                test_features = features[data_num[i-1]:]
                test_targets = targets[data_num[i-1]:]
                train_features = features[:data_num[i-1]]
                train_targets = targets[:data_num[i-1]]
                tuple = (train_features, train_targets, test_features, test_targets)
                TList.append(tuple)
        return TList

    raise NotImplementedError

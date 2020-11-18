import numpy as np


def num_class(dataset):
    classes_name = dataset.class_to_idx.keys()
    classes_idx = dataset.class_to_idx.values()
    classes_count = []
    for ii in classes_idx:
        a = len(np.where(np.array(dataset.targets) == ii)[0])
        classes_count.append(a)
    return dict(zip(classes_name, classes_count))


def train_valid_test(dataset, train_ratio, valid_ratio):
    train_ratio = train_ratio - valid_ratio
    dataset_num = len(dataset)
    train_num = int(train_ratio * dataset_num)
    valid_num = int(valid_ratio * dataset_num)
    indices = list(range(dataset_num))
    np.random.shuffle(indices)
    return indices[:train_num], indices[train_num: train_num+valid_num], indices[train_num+valid_num:]






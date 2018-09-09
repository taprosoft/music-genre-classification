
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_train_test_inds(y,train_proportion=0.7):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and
    testing sets are preserved (stratified sampling).
    '''

    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds, test_inds

def read_label_csv(label_csv):
    '''Read song names and corresponding label from csv file
    '''
    with open(label_csv, 'r') as f:
        songs = [line.strip().split(',') for line in f]
        song_names = [s[0] for s in songs]
        labels = [int(s[1]) - 1 for s in songs]

    return song_names, labels

def write_label_csv(song_names, labels, label_csv):
    '''Write song names and corresponding label to csv file
    '''
    with open(label_csv, 'w') as f:
        for s, l in zip(song_names, labels):
            f.write("{},{}\n".format(s, str(l + 1)))

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="train.csv", help='input file')
    parser.add_argument("--train_ratio", type=float, default=0.8, help='train ratio')
    args = parser.parse_args()

    song_names, labels = read_label_csv(args.input)
    song_names, labels = np.array(song_names), np.array(labels)
    train_inds, test_inds = get_train_test_inds(labels, train_proportion=args.train_ratio)

    dir_name = os.path.dirname(args.input)

    write_label_csv(song_names[train_inds], labels[train_inds], os.path.join(dir_name, 'train_set.csv'))
    write_label_csv(song_names[test_inds], labels[test_inds], os.path.join('val_set.csv'))


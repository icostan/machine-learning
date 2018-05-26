import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import feature_selection as fs


sns.set_style("whitegrid")


def plot_correlation_map(df):
    _, ax = plt.subplots(figsize=(16, 10))
    _ = sns.heatmap(
        df.corr(),
        cmap="RdYlGn",
        square=True,
        cbar=True, cbar_kws={'shrink': .8},
        ax=ax,
        annot=True, annot_kws={'fontsize': 12})


def plot_accuracy(history):
    # summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')


def plot_loss(history):
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()


def plot_univariate_scores(x, y, feature_names):
    chi_test, _ = fs.chi2(x, y)
    chi_test /= np.max(chi_test)

    f_test, _ = fs.f_classif(x, y)
    f_test /= np.max(f_test)

    mi_test = fs.mutual_info_classif(x, y)
    mi_test /= np.max(mi_test)

    feature_count = len(feature_names)

    plt.figure(figsize=(15, feature_count * 3))
    for i in range(feature_count):
        plt.subplot(feature_count / 2 + 1, 2, i + 1)
        plt.scatter(x.iloc[:, i], y)
        plt.xlabel(feature_names[i])
        plt.title("CHI={:.2f}, F={:.2f}, MI={:.2f}".format(
            chi_test[i], f_test[i], mi_test[i]))
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def boxplot(data, x, y):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x, y=y, data=data)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.show()

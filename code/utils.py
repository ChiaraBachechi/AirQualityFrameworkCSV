from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os.path
import calendar


def range_ARPA_NOx(val):
    clas = np.zeros(len(val), dtype=int)
    for i in np.arange(0, len(val)):
        if val[i] > 0 and val[i] <= 100:
            clas[i] = 1         #light_green
        else:
            if val[i] > 100 and val[i] <= 200:
                clas[i] = 2     #yellow
            else:
                if val[i] > 200 and val[i] <= 300:
                    clas[i] = 3         #orange
                else:
                    if val[i]>300 and val[i]<=400:
                        clas[i]=4       #red
                    else:
                        clas[i] =5    #violet
    return clas

def range_EEA_NOx(val):
    clas = np.zeros(len(val), dtype=int)
    for i in np.arange(0, len(val)):
        if val[i] > 0 and val[i] <= 50:
            clas[i] = 1     #dark_green
        else:
            if val[i] > 50 and val[i] <= 100:
                clas[i] = 2     #light_green
            else:
                if val[i] > 100 and val[i] <= 200:
                    clas[i] = 3     #yellow
                else:
                    if val[i] > 200 and val[i] <= 400:
                        clas[i] = 4     #orange
                    else:
                        clas[i] = 5         #red
    return clas
def range_ARPA_O3(val):
    clas = np.zeros(len(val), dtype=int)
    for i in np.arange(0, len(val)):
        if val[i] > 0 and val[i] <= 90:
            clas[i] = 1         #light_green
        else:
            if val[i] > 90 and val[i] <= 180:
                clas[i] = 2     #yellow
            else:
                if val[i] > 180 and val[i] <= 240:
                    clas[i] = 3         #orange
                else:
                    if val[i]>240 and val[i]<=300:
                        clas[i]=4       #red
                    else:
                        clas[i] =5    #violet
    return clas

def range_EEA_O3(val):
    clas = np.zeros(len(val), dtype=int)
    for i in np.arange(0, len(val)):
        if val[i] > 0 and val[i] <= 80:
            clas[i] = 1     #dark_green
        else:
            if val[i] > 80 and val[i] <= 100:
                clas[i] = 2     #light_green
            else:
                if val[i] > 100 and val[i] <= 120:
                    clas[i] = 3     #yellow
                else:
                    if val[i] > 120 and val[i] <= 140:
                        clas[i] = 4     #orange
                    else:
                        clas[i] = 5         #red
    return clas

def range_ARPA_CO(val):
    clas = np.zeros(len(val), dtype=int)
    for i in np.arange(0, len(val)):
        if val[i] > 0 and val[i] <= 5000:
            clas[i] = 1         #light_green
        else:
            if val[i] > 5000 and val[i] <= 10000:
                clas[i] = 2     #yellow
            else:
                if val[i] > 10000 and val[i] <= 15000:
                    clas[i] = 3         #orange
                else:
                    if val[i]>15000 and val[i]<=20000:
                        clas[i]=4       #red
                    else:
                        clas[i] =5    #violet
    return clas

def range_EEA_CO(val):
    clas = np.zeros(len(val), dtype=int)
    for i in np.arange(0, len(val)):
        if val[i] > 0 and val[i] <= 1000:
            clas[i] = 1     #dark_green
        else:
            if val[i] > 1000 and val[i] <= 5000:
                clas[i] = 2     #light_green
            else:
                if val[i] > 5000 and val[i] <= 10000:
                    clas[i] = 3     #yellow
                else:
                    if val[i] > 10000 and val[i] <= 15000:
                        clas[i] = 4     #orange
                    else:
                        clas[i] = 5         #red
    return clas





def plot_confusion_matrix(cm,
                          target_names,
                          month,
                          test,
                          pollutant,
                          dir,
                          scale,
                          model,
                          cmap=None,
                          normalize=False,
                          ):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(13, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix on '+ calendar.month_name[test])
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.autoscale()

    plt.savefig(os.path.join(dir, pollutant + '_' + str(calendar.month_name[month]) +'_'+str(calendar.month_name[test])+'_'+model+'_'+'_'+scale+ '.png'),dpi=96)


def plot_RMSE_ACC(reg,pollutant,month,months_ARPA,months_EEA,lag,model,dir):
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.figure(figsize=(10, 10))
    plt.barh(labels, reg)
    plt.ylabel("Months")
    plt.xlabel("RMSE")
    plt.title("Test on sensor " + str(dir) + " - month: " + str(calendar.month_name[month]))
    for i, v in enumerate(reg):
       plt.text(v + 3, i, str(round(v,1)), color='black')

    plt.savefig(os.path.join(str(dir) + "/RMSE",pollutant + '_RMSE_'+model+'_' + str(calendar.month_name[month]) + "_" + str(lag) + '.png'),dpi=96)
    plt.close()

    ind = np.arange(len(months_ARPA))
    width = 0.4

    fig, ax = plt.subplots()
    ax.barh(ind, months_EEA, width, color='red', label='EEA')
    ax.barh(ind + width, months_ARPA, width, color='green', label='ARPA')

    ax.set(yticks=ind + width, yticklabels=labels, ylim=[2 * width - 1, len(months_ARPA)])
    ax.set_xlabel('Train_size')
    ax.set_ylabel('RMSE')
    ax.legend()
    plt.title("Test on sensor " + str(dir) + " - month: " + str(calendar.month_name[month]))
    plt.savefig(os.path.join(dir + "/Accuracy",pollutant + '_ACCURACY_'+model+'_' + str(calendar.month_name[month]) + "_" + str(lag) + '.png'),dpi=96)
    plt.close('all')


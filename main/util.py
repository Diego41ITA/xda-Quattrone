import os
import sys
import numpy as np
import pandas as pd

def vecPredictProba(models, X):
    if type(X) is list:
        X = np.array(X)

    probas = np.empty((X.shape[0], len(models)))
    for i, model in enumerate(models):
        probas[:, i] = model.predict_proba(X)[:, 1]
    return probas


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def evaluateDataset(dataset, name):
    os.chdir(sys.path[0])
    os.chdir("../MDP_Dataset_Builder")
    np.save("./starting_combinations.npy", dataset)

    if os.name == "posix":
        os.system("./evaluate_adaptations.sh ./starting_combinations.npy")
    else:
        os.system("evaluate_adaptations.bat ./starting_combinations.npy")

    # Rename the file
    os.chdir("..")
    source_file = './MDP_Dataset_Builder/merge.csv'
    new_file = './results/' + name + '.csv'
    if os.path.exists(new_file):
        os.remove(new_file)
    # i = 1
    # while os.path.exists(new_file):
    #     new_file = './results/' + name + "(" + i + ")" + '.csv'
    os.rename(source_file, new_file)


def evaluateAdaptations(dataset, featureNames):

    customAdaptations = pd.DataFrame(dataset['custom_adaptation'].to_list(), columns=featureNames)
    nsga3Adaptations = pd.DataFrame(dataset['nsga3_adaptation'].to_list(), columns=featureNames)
    customAdaptations_anchors = pd.DataFrame(dataset['anchors_adaptation'].to_list(), columns=featureNames)
    customAdaptations_WIP = pd.DataFrame(dataset['wip_adaptation'].to_list(), columns=featureNames)

    evaluateDataset(customAdaptations, "customDataset")
    evaluateDataset(nsga3Adaptations, "nsga3Dataset")
    evaluateDataset(customAdaptations_anchors, "anchorsDataset")
    evaluateDataset(customAdaptations_WIP, "wipDataset")

def readFromCsv(path):
    results = pd.read_csv(path)
    columns = ["nsga3_adaptation", "custom_adaptation", "anchors_adaptation", "wip_adaptation", "nsga3_confidence", "custom_confidence", "anchors_confidence", "wip_confidence"]

    # numpy arrays are read as strings, must convert them back in arrays
    for c in columns:
        results[c] = results[c].apply(lambda x:  np.fromstring(x[1:-1], dtype=float, sep=' ')if x[1]!='[' else np.fromstring(x[2:-2], dtype=float, sep=' '))


    return results

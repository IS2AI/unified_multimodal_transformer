import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score


def EER_(cos_sim, labels):

    cos_sim = cos_sim.cpu().numpy()
    labels = labels.cpu().numpy()

    fpr, tpr, threshold = roc_curve(labels, cos_sim, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))] * 100
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    scores = cos_sim > eer_threshold

    return eer, scores

def EER_handmade(cos_sim, labels):

    thresholds = []
    FPR = []
    FNR = []

    for threshold in np.linspace(0.1,0.9,10):
        thresholds.append(threshold)
        scores = cos_sim > threshold

        FP = 0
        FN = 0
        TN = 0
        TP = 0

        for i in range(len(scores)):
            if scores[i]:
                if scores[i] == labels[i]:
                    TP += 1
                else:
                    FP += 1
            else:
                if scores[i] == labels[i]:
                    TN += 1
                else:
                    FN += 1

        fpr = FP / (FP + TN)
        fnr = FN / (FN + TP)

        FPR.append(fpr)
        FNR.append(fnr)
        
    FNR = np.array(FNR)
    FPR = np.array(FPR)
    EER_threshold = thresholds[np.nanargmin(np.absolute((FNR - FPR)))]
    EER = FPR[np.nanargmin(np.absolute((FNR - FPR)))]

    return EER, EER_threshold

def accuracy_(labels, scores):
    errors = np.absolute(scores-labels.numpy()).sum()
    correct = len(labels) - errors
    return correct / len(labels) * 100

def accuracy_sklearn(labels, scores):
    return accuracy_score(labels, scores)
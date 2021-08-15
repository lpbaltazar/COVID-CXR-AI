import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, \
							recall_score, f1_score, classification_report

from imblearn.metrics import geometric_mean_score as gmean

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300

def true_positives(y_true, y_pred):
    
    y_t = K.batch_flatten(y_true)
    y_p = K.batch_flatten(y_pred)
    
    return K.sum(y_t*y_p)


def true_negatives(y_true, y_pred):
    
    y_t = K.batch_flatten(1-y_true)
    y_p = K.batch_flatten(1-y_pred)

    return K.sum(y_t*y_p)


def ppv(y_true, y_pred):
    return true_positives(y_true, y_pred) / (K.sum(y_true) + K.epsilon())


def npv(y_true, y_pred):
    return true_negatives(y_true, y_pred) / (K.sum(1-y_true)+K.epsilon())


def get_conf_matrix_per_class(y_true, y_pred, class_names):
	conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
	TP = np.diag(conf_matrix)
	FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
	FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
	TN = conf_matrix.sum() - (FP + FN + TP)
	return TP, FN, FP, TN


def get_metrics(y_true, y_pred, class_names, name_id_map):
	TP, FN, FP, TN = get_conf_matrix_per_class(y_true, y_pred, class_names)
	sensitivity = TP / (TP + FN)
	specificity = TN / (TN + FP)
	ppv = TP / (TP + FP)
	npv = TN / (TN + FN)
	likelihood_pos = sensitivity / (1 - specificity)
	likelihood_neg = (1 - sensitivity) / specificity

	scores = [sensitivity, specificity, ppv, npv, TP, FP, FN, TN, likelihood_pos, likelihood_neg]

	scores = pd.DataFrame(scores).T
	scores.reset_index(inplace = True)
	scores.rename(columns = {	0:"sensitivity", 
	                            1:"specificity", 
	                            2:"ppv", 
	                            3:"npv",
	                            4:"true_positive",
	                            5:"false_positive",
	                            6:"false_negative",
	                            7:"true_negative",
	                            8:"lr(+)",
	                            9:"lr(-)",
	                            "index":"label"}, inplace = True)
	
	scores["label"] = scores["label"].apply(lambda x: name_id_map[x])

	acc = accuracy_score(y_true, y_pred)
	ppv = precision_score(y_true, y_pred, average = "weighted")
	sn = recall_score(y_true, y_pred, average = "weighted")
	f1 = f1_score(y_true, y_pred, average = "weighted")
	gm = gmean(y_true, y_pred, average = "weighted")

	display("================== GENERAL PERFORMANCE ==================")
	display(f"Accuracy: {round(acc, 2)}")
	display(f"PPV: {round(ppv,2)}")
	display(f"Sensitivity: {round(sn,2)}")
	display(f"F1-Score: {round(f1,2)}")
	display(f"G-mean: {round(gm,2)}")
	display("=========================================================")
	
	display("================= PER LABEL PERFORMANCE =================")
	display(scores[["label", "sensitivity", "specificity", "ppv", "npv"]])
	display("=========================================================")


def plot_confmatrix(y_true, y_pred):
	conf_matrix = confusion_matrix(y_true, y_pred, labels=range(4))
	label = ["normal", "covid", "viral", "bacterial"]
	fig, ax = plt.subplots()
	im = ax.imshow(conf_matrix, cmap = "Blues")
	ax.set_xticks(np.arange(len(label)))
	ax.set_yticks(np.arange(len(label)))
	ax.set_xticklabels(label)
	ax.set_yticklabels(label)
	for i in range(len(label)):
		for j in range(len(label)):
			text = ax.text(j, i, conf_matrix[i, j],
							ha="center", va="center", color = "w")
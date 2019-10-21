import pandas as pd
import sys
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix

threshold = 0.9
filename  = 'all.csv'

if(len(sys.argv) == 2):
    threshold = float(sys.argv[1])

if(len(sys.argv) == 3):
    threshold = float(sys.argv[1])
    filename  = sys.argv[2]

result = pd.read_csv(filename, sep=',')

y    = result['y']
loss = result['loss']

fpr, tpr, _ = roc_curve(y, loss)

roc_auc = auc(fpr, tpr)

print("roc_auc: {}".format(roc_auc))

y_pred_bool = (loss > threshold)
print(classification_report(y, y_pred_bool, digits=4))
print(confusion_matrix(y, y_pred_bool))

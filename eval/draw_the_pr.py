import os
from matplotlib import pyplot as plt
import numpy as np
# from kitti import options_clip
from sklearn import metrics
import sys

sys.path.append(os.getcwd())



# opt = options_clip.Options()

sequence = '00'

gt_db = np.load(os.path.join('1_10_00','00_gt_db.npy'))
pred_db = np.load(os.path.join('1_10_00','00_DL_db.npy'))

precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)
# plot p-r curve
plt.figure(1)
lw = 2
plt.plot(recall, precision, color='darkorange',
            lw=lw, label='base') 
plt.axis([0,1,0,1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('DL Precision-Recall Curve')
plt.legend(loc="lower right")

gt_db = np.load(os.path.join('1_20_00','00_gt_db.npy'))
pred_db = np.load(os.path.join('1_20_00','00_DL_db.npy'))
precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)
plt.plot(recall, precision, color='red',
            lw=lw, label='base_semantic') 
plt.legend(loc="lower right")

gt_db = np.load(os.path.join('1_30_00','00_gt_db.npy'))
pred_db = np.load(os.path.join('1_30_00','00_DL_db.npy'))

precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)
plt.plot(recall, precision, color='green',
            lw=lw, label='base_semantic_graph') 
plt.legend(loc="lower right")


pr_out = os.path.join( "curve.png")
plt.savefig(pr_out)
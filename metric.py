# https://github.com/a-nagrani/VoxSRC2020/tree/master

from sklearn.metrics import *
from operator import itemgetter
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_eer(truth_labels, scores, pos=1):
	# truth_labels denotes groundtruth scores,
	# scores denotes the prediction scores.

	try:
		fpr, tpr, thresholds = roc_curve(truth_labels, scores, pos_label=pos)
		eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
		thresh = float(interp1d(fpr, thresholds)(eer))
	
	except:
		if sum(truth_labels) == 0:
			eer = 0
			thresh = None

		elif sum(truth_labels) == len(truth_labels):
			eer = 1
			thresh = None

	return eer, thresh

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def compute_error_rates(truth_label, scores):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      truth_label = [truth_label[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(truth_label)):
          if i == 0:
              fnrs.append(truth_label[i])
              fprs.append(1 - truth_label[i])
          else:
              fnrs.append(fnrs[i-1] + truth_label[i])
              fprs.append(fprs[i-1] + 1 - truth_label[i])
      fnrs_norm = sum(truth_label)
      fprs_norm = len(truth_label) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def compute_min_dcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold
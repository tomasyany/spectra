"""This class will perform different cross validations techniques."""

from sklearn.cross_validation import KFold
from numpy import sqrt


class Score(object):

    def __init__(self, actual_class):
        self.actual_class = actual_class

        self.TP = 0 #True Positive
        self.FP = 0 #False Positive
        self.TN = 0 #True Negative
        self.FN = 0 #False Negative

        self.TP_part = 0 #True Positive %
        self.FP_part = 0 #False Positive %
        self.TN_part = 0 #True Negative %
        self.FN_part = 0 #False Negative %

        self.total_values = 0

    @staticmethod
    def k_cross(data_size, n_folds, shuffle=True):

        kf = KFold(data_size, n_folds, shuffle=shuffle)
        for train_set, test_set in kf:
                yield (train_set, test_set)

    # Function for setting values of the matrix to 0
    def go_zero(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

        self.TP_part = 0
        self.FP_part = 0
        self.TN_part = 0
        self.FN_part = 0

        self.total_values = 0

    def update_confusion_matrix(self, predicted_class, real_class):
        if (predicted_class == self.actual_class and
           predicted_class == real_class):
            self.TP += 1
        elif (predicted_class == self.actual_class and
             predicted_class != real_class):
            self.FP += 1
        elif (predicted_class != self.actual_class and
             real_class != self.actual_class):
            self.TN+= 1
        elif (predicted_class != self.actual_class and
             real_class == self.actual_class):
            self.FN += 1

        self.total_values += 1.0
        self.update_parts()

    # Updating the pourcetange
    def update_parts(self):
        self.TP_part = round(self.TP/self.total_values*100,3)
        self.FP_part = round(self.FP/self.total_values*100,3)
        self.TN_part = round(self.TN/self.total_values*100,3)
        self.FN_part = round(self.FN/self.total_values*100,3)

    def compute_recall(self):
        if self.TP == 0:
            return 0
        return self.TP / (self.TP + self.FN)

    def compute_specificity(self):
        if self.TN == 0:
            return 0
        return self.TN / (self.FP + self.TN)

    def compute_precision(self):
        if self.TP == 0:
            return 0
        return self.TP / (self.TP + self.FP)

    def compute_negative_predictive_value(self):
        if self.TN == 0:
            return 0
        return self.TN / (self.TN + self.FN)

    def compute_fall_out(self):
        if self.FP == 0:
            return 0
        return self.FP / (self.FP + self.TN)

    def compute_false_discovery_rate(self):
        if self.FP == 0:
            return 0
        return self.FP / (self.FP + self.TP)

    def compute_false_negative_rate(self):
        if self.FN == 0:
            return 0
        return self.FN / (self.FN + self.TP)

    def compute_accuracy(self):
        if self.TP + self.TN == 0:
            return 0
        return (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)

    def compute_fscore(self):
        if self.TP == 0:
            return float(0)
        return float(2 * self.TP / (2 * self.TP + self.FP + self.FN))

    def compute_MCC (self):
        # Returns the Matthews correlation coefficient
        if (self.TP * self.TN - self.FP * self.FN) == 0:
            return 0
        return (self.TP * self.TN - self.FP * self.FN) / \
            sqrt((self.TP + self.FP) *
                     (self.TP + self.FN) *
                     (self.TN + self.FP) *
                     (self.TN + self.FN))

    def get_results(self):
        results = {}
        results['3_recall'] = self.compute_recall()
        results['4_specificity'] = self.compute_specificity()
        results['5_precision'] = self.compute_precision()
        results['6_NPV'] = self.compute_negative_predictive_value()
        results['7_fall_out'] = self.compute_fall_out()
        results['8_FDR'] = self.compute_false_discovery_rate()
        results['9_FNR'] = self.compute_false_negative_rate()

        results['1_accuracy'] = self.compute_accuracy()
        results['0_fscore'] = self.compute_fscore()
        results['2_MCC'] = self.compute_MCC()

        return results

# Sources: Fawcett (2006) and Powers (2011).

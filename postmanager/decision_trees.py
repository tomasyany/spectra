"""This class contains the decision trees implementation. """

from sklearn.tree import DecisionTreeClassifier
from bisect import bisect

class DecisionTree(object):

    def __init__(self,
                 min_red,
                 max_red,
                 real_peaks,
                 train_features,
                 train_labels):

        self.min_red = min_red
        self.max_red = max_red
        self.real_intervals = [(i * (1 + min_red), i * (1 + max_red))
                            for i in real_peaks]

        self.train_features = train_features
        self.train_labels = train_labels

        self.tree = DecisionTreeClassifier()

    @staticmethod
    def place_point_in_range(point, intervals_endpoints):
        """Returns, if found, the interval where the point belongs."""
        i = bisect(intervals_endpoints, point)

        if i % 2 == 0:
            if point == intervals_endpoints[i - 1]:
                return (intervals_endpoints[i - 1], intervals_endpoints[i])
            else:
                return None
        else:
            return (intervals_endpoints[i - 1], intervals_endpoints[i])

    def get_data (self, features_list):
        """Returns the array of size [n_samples, n_features] with the features
        per sample.
        Must transform the features in the form: `EW in range I equals...`"""

        data = []
        intervals_endpoints = []
        for interval in self.real_intervals:
            intervals_endpoints.extend(interval)

        for curve_features in features_list:
            # A curve_features is a list of features
            curve_data = [0] * len(self.real_intervals)
            for feature in curve_features:
                # A feature is a tuple like (lambda_obs, equi_width)
                # lambda_obs corresponds to the lambda where the peak was
                # observed
                lambda_obs = feature[0]
                equi_width = feature[1]

                found_interval = DecisionTree.place_point_in_range(
                                              lambda_obs,
                                              intervals_endpoints)
                if found_interval:
                    interval_index = self.real_intervals.index(found_interval)
                    curve_data[interval_index] = equi_width

            data.append(curve_data)

        return data

    def fit_tree(self):
        """Fits the tree to the relevant data and labels."""

        data = self.get_data(self.train_features)
        labels = self.train_labels

        return self.tree.fit(data, labels)

    def predict_test_classes(self, test_features):
        """Returns the predicted classes for the testing set."""

        test_features = self.get_data(test_features)
        predicted_classes = self.tree.predict(test_features)

        return predicted_classes

    def predict_test_classes_proba(self, test_features):
        """Returns the predicted classes probabilities for the testing set."""

        test_features = self.get_data(test_features)
        predicted_classes = self.tree.predict_proba(test_features)

        return predicted_classes

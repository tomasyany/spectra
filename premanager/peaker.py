"""This module contains the class for differents peak detection approaches."""

from numpy import mean
from numpy import std
from numpy import amax
from numpy import floor
from numpy import ones
from numpy import sum
from numpy import float
from numpy import concatenate
from numpy import zeros
from numpy import append
from numpy import array
from numpy import exp
from numpy import sqrt
from numpy import pi
from numpy import log

from premanager.extractor import Extractor

class Peaker(object):

    def __init__(self, X):
        self.X  = X
        self.N = len(X)


    def n_plus(self, k, i):
        """Returns the k right temporals neighbours of self.X."""

        if i == self.N - 1:
            return zeros(self.N)

        elif i + k > self.N:
            return self.X[i + 1 :]

        else:
            return self.X[i + 1 : i + k + 1]


    def n_minus(self, k, i):
        """Returns the k left temporals neighbours of self.X."""

        if i == 0:
            return zeros(self.N)

        elif i - k < 0:
            return self.X[: i - 1]

        else:
            return self.X[i - k : i]

    def n_plus_BIS(self, k, i):
        """Returns the k right temporals neighbours of self.X."""

        if i == self.N - 1:
            return zeros(self.N)

        elif i + k > self.N:
            return self.X[i + k + 1 :]

        else:
            return self.X[i + k + 1 : i + 2*k + 1]


    def n_minus_BIS(self, k, i):
        """Returns the k left temporals neighbours of self.X."""

        if i == 0:
            return zeros(self.N)

        elif i - k < 0:
            return self.X[: i - k - 1]

        else:
            return self.X[i - 2*k : i - k]

    def n_around(self, k, i):
        """Returns the concatenation of n_minus and n_plus."""

        return concatenate((self.n_minus(k,i), self.n_plus(k,i)))


    def n_total(self, k, i):
        """Returns the concatenation of n_minus and n_plus."""

        return self.X[i - k : i + k + 1]


    def gauss_kernel(x):
        return exp(-(x)**2 / 2) / sqrt(2 * pi)


    def epane_kernel(x):
        if abs(x) < 1:
            return 3 / 4 * (1 - x**2)
        else:
            return 0


    def parzen_window(self, i, w, set_A, kernel=None):
        M = len(set_A)
        if i + w >= M:
            i = i - w
        kernel = Peaker.gauss_kernel
        aux_sum = 0

        for j in range (1, M):
            aux_term = float((set_A[i] - set_A[j]) /
                       abs(set_A[i] - set_A[i+w]))
            aux_sum += kernel(aux_term)

        return float(aux_sum / (M * abs(set_A[i] - set_A[i+w])))


    def entropy(self, w, set_A):
        aux_sum = 0
        M = len(set_A)

        for i in range(1, M):
            p_w_i = self.parzen_window(i, w, set_A)
            aux_sum += -1 * p_w_i * log(p_w_i)

        return aux_sum

    def S_ONE(self, k, i):
        """Computes the S1 peak function."""

        x_i_array = self.X[i] * ones(k)

        left_max = amax(x_i_array - self.n_minus(k, i))
        right_max = amax(x_i_array - self.n_plus(k, i))

        return float((left_max + right_max) / 2)


    def S_TWO(self, k, i):
        """Computes the S2 peak function."""

        x_i_array = self.X[i] * ones(k)-1

        left_avg = float(sum(x_i_array - self.n_minus(k, i)) / k)
        right_avg = float(sum(x_i_array - self.n_plus(k, i)) / k)

        return float((left_avg + right_avg) / 2)

    def S_THREE(self, k, i):
        """Computes the S3 peak function."""

        left_avg = self.X[i] - float(sum(self.n_minus(k, i)) / k)
        right_avg = self.X[i] - float(sum(self.n_plus(k, i)) / k)

        return float((left_avg + right_avg) / 2)

    def S_THREE_BIS(self, k, i):
        """Computes the S3 peak function."""

        left_avg = self.X[i] - float(sum(self.n_minus_BIS(k, i)) / k)
        right_avg = self.X[i] - float(sum(self.n_plus_BIS(k, i)) / k)

        return float((left_avg + right_avg) / 2)

    def S_FOUR(self, k, i, w=10):
        """Computes the S4 peak function."""

        return (self.entropy(w, self.n_around(k, i)) -
               self.entropy(w, self.n_total(k, i)))


    def S_FIVE(self, k, i, t=3):
        """Computes the S5 peak function."""

        n_mean = mean(self.n_around(k,i))
        n_std = std(self.n_around(k,i))

        if self.X[i] > n_mean and abs(self.X[i] - n_mean) >= t * n_std:
            return True
        else:
            return False

    def get_peak_function(self, peak_function_type):
          return [
                self.S_ONE,
                self.S_TWO,
                self.S_THREE,
                self.S_FOUR,
                self.S_FIVE,
                self.S_THREE_BIS
            ][peak_function_type - 1]

    def get_peaks(self, peak_function_type, k, h=1):
        """Returns all the peaks for the given peak function and threshold h.
        The peaks are given in the form of their indices in the original
        vector.
        k is the half neighbours window size."""

        peaks = array([], int)
        score = zeros(self.N)

        peak_function = self.get_peak_function(peak_function_type)

        for i in range(k, self.N - k):
            score[i] = peak_function(k, i)

        # Mean and std only for positive values
        scores_mean = mean(score[score>0])
        scores_std = std(score[score>0])

        if peak_function.__name__ != 'S_FIVE':
            for i in range(1, self.N):
                if score[i] > 0 and (score[i]-scores_mean) > h * scores_std:
                   peaks = append(peaks, i)
        else:
            for i in range(1, self.N):
                if score[i] > 0:
                   peaks = append(peaks, i)

        filtered_peaks = self.filter_peaks(peaks, k)

        return filtered_peaks

    def curvature_filtering(self, peaks):
        """Doesn't consider peak if curvature doesn't change."""

        peaks_to_remove = []

        for peak in peaks:
            peak_value = self.X[peak]

            left_neighbour_value = self.X[peak - 1]
            right_neighbour_value = self.X[peak + 1]

            left_curvature = peak_value - left_neighbour_value
            right_curvature = right_neighbour_value - peak_value

            if not(left_curvature >= 0 and right_curvature <=0):
                peaks_to_remove.append(peak)

        new_peaks = [i for i in peaks if i not in peaks_to_remove]
        return new_peaks

    def window_filtering(self, peaks, window):
        """Doesn't consider peak if more than 1 per window."""

        peaks_to_remove = []

        for peak_i in peaks:
            for peak_j in peaks:

                if peak_i != peak_j and abs(peak_j - peak_i) < window:
                    if self.X[peak_i] <= self.X[peak_j]:
                        peaks_to_remove.append(peak_i)
                    else:
                        peaks_to_remove.append(peak_j)

        new_peaks = [i for i in peaks if i not in peaks_to_remove]

        return new_peaks

    def zero_filtering(self, peaks):
        """Doesn't consider peak if y_obs == 0."""

        peaks_to_remove = []

        for peak in peaks:
            if self.X[peak] == 0:
                peaks_to_remove.append(peak)

        new_peaks = [i for i in peaks if i not in peaks_to_remove]

        return new_peaks

    def filter_peaks(self, peaks, window):
        """Selects only the true peaks among the found ones."""

        new_peaks = self.window_filtering(peaks, window)
        new_peaks = self.curvature_filtering(new_peaks)
        new_peaks = self.zero_filtering(new_peaks)

        return new_peaks

    def compare_original_detected(self, file_name, fitsFile, original_peaks,
                                  detected_peaks, peak_score, lambda_window):

        """Compare the original peaks with the found ones for the given
        spectrum. Updates peak_score confusion matrix."""

        nw = Extractor.emit_lambda_window_to_neighbours(
                       file_name, fitsFile, lambda_window)
        nw = int(nw / 2)
        original_ranges = [[i-nw, i+nw] for i in original_peaks]

        for detected in detected_peaks:
            for original_range in original_ranges:

                if original_range[0] <= detected <= original_range[1]:
                    found_peak = True

                    # TP += 1
                    peak_score.update_confusion_matrix(True, True)

                    found_index = original_ranges.index(original_range)

                    for i in range(0, found_index):
                        # FN += 1
                        peak_score.update_confusion_matrix(False, True)

                    original_ranges = original_ranges[found_index+1:]
                    break
                else:
                    found_peak = False

            if not found_peak:
                # FP += 1
                peak_score.update_confusion_matrix(True, False)

        return None


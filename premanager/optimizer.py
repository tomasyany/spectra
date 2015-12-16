"""This class optimizes the score functions parameters."""

from itertools import product
from numpy.random import randint

from premanager.peaker import Peaker
from premanager.extractor import Extractor
from postmanager.score import Score


class Optimizer(object):

    def __init__(self, all_file_names, peak_function,
                 left_cut_point, right_cut_point, lambda_window):

        self.all_file_names = all_file_names
        self.peak_function = peak_function
        self.left_cut_point = left_cut_point
        self.right_cut_point = right_cut_point
        self.lambda_window = lambda_window

        self.all_param_scores = {}

    @staticmethod
    def uniform_grid_generator(param_min, param_steps, amount_points):
        """Generates a uniformly distributed grid.
        param_min is a list of the minimums for each param.
        param_steps is the step for each parameters (same order as param_min).
        amount_points is the desired amount of points for each param."""

        dim = len(param_min)
        aux_list = []

        for i in range(dim):
            aux_list.append([])
            for step_i in range(amount_points[i]):
                aux_list[i].append(param_min[i] + param_steps[i] * step_i)

        grid = [item for item in product(*aux_list)]

        return grid

    @staticmethod
    def random_grid_generator(param_min, param_max, amount_points):
        """Generates a uniform randomly distributed grid.
        param_min is a list of the minimums for each param.
        param_max is a list of the maximums for each param.
        amount_points is the desired amount of points for each param."""

        dim = len(param_min)
        aux_list = []

        for i in range(dim):
            aux_list.append(randint(param_min[i], param_max[i],
                            amount_points[i]))

        grid = [item for item in product(*aux_list)]

        return grid

    def grid_search(self, param_min, param_steps, amount_points):
        """Returns the best parameters with a grid search
           for the self.peak_function."""

        param_tuples = Optimizer.uniform_grid_generator(
                       param_min, param_steps, amount_points)

        return self.compute_search(param_tuples)

    def random_search(self, param_min, param_max, amount_points):
        """Returns the best parameters with a random search
       for the self.peak_function."""

        param_tuples = Optimizer.random_grid_generator(
                       param_min, param_max, amount_points)

        return self.compute_search(param_tuples)

    def compute_search(self, param_tuples):
        """Returns the best tuple for the param_tuples argument."""

        avg_param_score = self.score_computation(param_tuples)

        best_tuple = max(avg_param_score, key=avg_param_score.get)

        return best_tuple


    def curves_scores_computation(self, all_param_tuples):
        """Computes the score for each param tuple and file."""

        current_file_index=0
        total_files = len(self.all_file_names)
        for file_name in self.all_file_names:

            fitsFile = Extractor.open_fits_file(file_name)

            galaxy_type = file_name.split('/')[1]

            x_emit, values = Extractor.get_emit_values(
                                              file_name,
                                              fitsFile,
                                              self.left_cut_point,
                                              self.right_cut_point)


            original_peaks = Extractor.get_original_peaks(
                             file_name, fitsFile, galaxy_type,
                             x_emit)

            pos_peaker = Peaker(values)

            neg_values = [-i for i in values]
            neg_peaker = Peaker(neg_values)

            file_score = Score(True)

            for i in range(0, len(all_param_tuples)):
                param_tuple = all_param_tuples[i]

                pos_peaks = pos_peaker.get_peaks(self.peak_function,
                        *param_tuple)
                neg_peaks = neg_peaker.get_peaks(self.peak_function,
                        *param_tuple)

                detected_peaks = []
                detected_peaks.extend(pos_peaks)
                detected_peaks.extend(neg_peaks)

                # Detected peaks here
                detected_peaks.sort()

                pos_peaker.compare_original_detected(file_name,
                                                     fitsFile,
                                                     original_peaks,
                                                     detected_peaks,
                                                     file_score,
                                                     self.lambda_window)

                score = file_score.compute_fscore()

                # Here it saves it to a dictionary
                if param_tuple in self.all_param_scores:
                    self.all_param_scores[param_tuple].append(score)
                else:
                    self.all_param_scores[param_tuple] = [score]


                file_score.go_zero()

            Extractor.close_fits_file(fitsFile)

            current_file_index += 1


    def score_computation(self, all_param_tuples):
        """Return the average F1 score for all the param tuples among the
        curves."""

        self.curves_scores_computation(all_param_tuples)

        avg_param_score = {}
        for param_tuple, scores in self.all_param_scores.items():
            avg_param_score[param_tuple] = sum(scores)/float(len(scores))

        return avg_param_score


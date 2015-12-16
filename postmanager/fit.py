"""This class computes the gaussian fitting of a point and the equivalent
width of a peak."""

from numpy import sqrt
from numpy import pi
from numpy import exp
from numpy import linspace
from numpy import mean
from numpy import array

from scipy.optimize import curve_fit
from scipy.integrate import quad

from premanager.extractor import Extractor


class Fit(object):

    def __init__(self, fitsFile, x_obs, values, peaks, window):
        self.x_obs = x_obs
        self.values = values
        self.peaks = peaks # List of indices of x_obs
        self.window = window # In Angstrom
        self.window_neighbours = Extractor.obs_lambda_window_to_neighbours(
                                      fitsFile, window)
        self.obs_delta_lambda = Extractor.get_obs_delta_lambda(fitsFile)

    @staticmethod
    def gaussian(x, amp, cen, wid):
        return amp * exp(-(x-cen)**2 /(2 * wid**2))

    def fit_gauss(self, peak_index, x, y):
        """Fits (x,y) to a gaussian profile around the peak."""

        amp0 = self.values[peak_index]
        cen0 = self.x_obs[peak_index]
        wid0 = self.window

        init_param = [amp0, cen0, wid0] # Initial parameters
        best_param, covar = curve_fit(Fit.gaussian, x, y, p0 = init_param)

        return best_param

    def get_equivalent_width(self, peak_index):
        """Returns the equivalent width for the given peak."""

        left_neigh = peak_index - int(self.window_neighbours/2)
        right_neigh = peak_index + int(self.window_neighbours/2)


        x = self.x_obs[left_neigh : right_neigh]
        y = self.values[left_neigh : right_neigh]

        values_sides = []
        values_sides.extend(self.values[peak_index -\
                            self.window_neighbours + 1 : left_neigh])
        values_sides.extend(self.values[right_neigh :\
                            peak_index + self.window_neighbours])

        continuum_level = mean(values_sides)

        try:
            best_param = self.fit_gauss(peak_index, x, y)

        except RuntimeError:
            # Compute integral with sum
            integral = self.obs_delta_lambda *\
                    sum([continuum_level - value for value in y])
            best_param = array([0,0,0])

        else:
            closest_index = self.x_obs.index(
                    min(self.x_obs, key=lambda s:abs(s-best_param[1])))

            index_corr= peak_index - closest_index

            x = self.x_obs[left_neigh - index_corr + 1 :
                           right_neigh - index_corr + 1]

            y = self.values[left_neigh - index_corr + 1 :
                           right_neigh - index_corr + 1]

            inf_limit = self.x_obs[closest_index] - self.window/2
            sup_limit = self.x_obs[closest_index] + self.window/2

            integrand = lambda s: continuum_level - Fit.gaussian(s, *best_param)
            integral, err = quad(integrand, inf_limit, sup_limit)


        equivalent_width = float(integral / continuum_level)

        return equivalent_width
        # return x, y, best_param, equivalent_width, continuum_level

    def get_all_EW(self):
        """Returns a list with all the EW widths for all the peaks."""

        all_EW = []
        for peak_index in self.peaks:
            all_EW.append((self.x_obs[peak_index],
                    self.get_equivalent_width(peak_index)))

        return all_EW


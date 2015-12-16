"""Interface for importing files and premanaging them."""

import os
from astropy.io import fits

# Definition of emission and absorption lines

ALL_LINES = [2800, 2850, 3096, 3360, 3727, 3833, 3934, 3968, 4102, 4308, 4341,
             4383, 4861, 4959, 5007, 5183, 5270, 5335]

# PASSIVE_LINES = {3833, 3934, 3968, 4308}
PASSIVE_INDEXES = [5, 6, 7, 9]
PASSIVE_LINES = [ALL_LINES[i] for i in PASSIVE_INDEXES]

# POST_LINES = {3833, 3934, 3968, 4959}
POST_INDEXES = [5, 6, 7, 13]
POST_LINES = [ALL_LINES[i] for i in POST_INDEXES]

# QUIESCENT_LINES ={3727, 3833, 3934, 3968}
QUIESCENT_INDEXES = [4, 5, 6, 7]
QUIESCENT_LINES = [ALL_LINES[i] for i in QUIESCENT_INDEXES]

# DUSTY_LINES = {3727, 3833, 5007}
DUSTY_INDEXES = [4, 5, 14]
DUSTY_LINES = [ALL_LINES[i] for i in DUSTY_INDEXES]

# STARBURST_LINES = {3728,4861,4959,5007}
STARBURST_INDEXES = [4, 12, 13, 14]
STARBURST_LINES = [ALL_LINES[i] for i in STARBURST_INDEXES]

GALAXIES_LINES = {'PASSIVE': PASSIVE_LINES,
                  'POST': POST_LINES,
                  'QUIESCENT': QUIESCENT_LINES,
                  'DUSTY': DUSTY_LINES,
                  'STARBURST': STARBURST_LINES
                  }

SELECTED_LINES_UNION = set(PASSIVE_LINES) |\
                       set(POST_LINES) |\
                       set(QUIESCENT_LINES) |\
                       set(DUSTY_LINES) |\
                       set(STARBURST_LINES)

ALL_SELECTED_LINES = list(SELECTED_LINES_UNION)
ALL_SELECTED_LINES.sort()


class Extractor(object):
    """Extracts information and data from the fits file."""

    @staticmethod
    def open_fits_file(file_name):
        return fits.open(file_name, memmap=False)

    @staticmethod
    def close_fits_file(fitsFile):
        fitsFile.close()

    @staticmethod
    def get_header(fitsFile):
        return fitsFile['PRIMARY'].header

    @staticmethod
    def get_lambda_zero(fitsFile):
        return Extractor.get_header(fitsFile)['CRVAL1']

    @staticmethod
    def get_obs_delta_lambda(fitsFile):
        return Extractor.get_header(fitsFile)['CD1_1']

    @staticmethod
    def get_emit_delta_lambda(file_name, fitsFile):
        redshift = Extractor.get_redshift(file_name)
        delta_lambda = Extractor.get_obs_delta_lambda(fitsFile)

        return delta_lambda/(1+redshift)

    @staticmethod
    def get_values(fitsFile):
        original_values = fitsFile['PRIMARY'].data
        # We perform a change in the scale for plotting reasons
        # No side effects observed
        values = [1e18*i if i>0 else 0 for i in original_values]

        return values

    @staticmethod
    def get_redshift(file_name):
        with open('inputs/_redshifts.tab', 'r') as f:
            contents = [x.strip().split('\t') for x in f]
        for fn in contents:
            sf = file_name.split('/')[2].replace('.fits', '')
            if fn[0] == sf:
                return float(fn[1])
        print('Redshift not found.')

    @staticmethod
    def obs_lambda_window_to_neighbours(fitsFile, lambda_window):
        """Tranform the lambda window into neighbours window."""

        obs_delta_lambda = Extractor.get_obs_delta_lambda(fitsFile)
        return int(round(lambda_window / obs_delta_lambda))

    @staticmethod
    def emit_lambda_window_to_neighbours(file_name, fitsFile, lambda_window):
        """Tranform the emit lambda window into neighbours window."""

        emit_delta_lambda = Extractor.get_emit_delta_lambda(file_name, fitsFile)
        return int(round(lambda_window / emit_delta_lambda))

    @staticmethod
    def get_original_peaks(file_name, fitsFile, galaxy_type,
                           x_emit):
        """Get the original peaks from the spectrum."""
        peaks = []

        galaxy_lines = GALAXIES_LINES[galaxy_type]
        for line in galaxy_lines:
            closest_index = x_emit.index(min(x_emit, key=lambda x:abs(x-line)))
            if not (closest_index == 0 or closest_index == len(x_emit) - 1):
                peaks.append(closest_index)

        return peaks

    @staticmethod
    def get_obs_values(file_name, fitsFile,
                        left_cut_point, right_cut_point):
        """Returns the x_obs and the values from the fitsFile."""

        values = Extractor.get_values(fitsFile)

        l_zero = Extractor.get_lambda_zero(fitsFile)
        obs_delta_lambda = Extractor.get_obs_delta_lambda(fitsFile)

        x_obs = [l_zero + i*obs_delta_lambda for i in range(0, len(values))]

        closest_left_cut_point = min(
                                 x_obs, key=lambda x: abs(x-left_cut_point))
        closest_right_cut_point = min(
                                  x_obs, key=lambda x: abs(x-right_cut_point))

        left_cut_point_index = x_obs.index(closest_left_cut_point)
        right_cut_point_index = x_obs.index(closest_right_cut_point)

        # Correct x_obs and values
        x_obs = x_obs[left_cut_point_index : right_cut_point_index]
        new_values = values[left_cut_point_index : right_cut_point_index]

        return x_obs, new_values

    @staticmethod
    def get_emit_values(file_name, fitsFile,
                        left_cut_point, right_cut_point):
        """Returns the x_emit and the values from the fitsFile."""

        x_obs, values = Extractor.get_obs_values(file_name, fitsFile,
                                  left_cut_point, right_cut_point)

        redshift = Extractor.get_redshift(file_name)
        x_emit = [i/(1 + redshift) for i in x_obs]

        return x_emit, values

    @staticmethod
    def get_file_class_string(file_name):
        """Get the class of the galaxy by its name."""

        return file_name.split('/')[1]

    @staticmethod
    def get_file_class_number(file_name, class_string=None):
        """Get the class of the galaxy by its number."""
        if class_string == None:
            class_string = Extractor.get_file_class_string(file_name)

        if class_string == "PASSIVE":
            return 0
        elif class_string == "POST":
            return 1
        elif class_string == "QUIESCENT":
            return 2
        elif class_string == "DUSTY":
            return 3
        elif class_string == "STARBURST":
            return 4

    @staticmethod
    def get_class_string_from_number(class_number):
        """Transform the class of the galaxy from number to string."""

        if class_number == 0:
            return "PASSIVE"
        elif class_number == 1:
            return "POST"
        elif class_number == 2:
            return "QUIESCENT"
        elif class_number == 3:
            return "DUSTY"
        elif class_number == 4:
            return "STARBURST"

    @staticmethod
    def get_all_redshifts():
        """Returns all the redshifts from the classified inputs."""

        with open('inputs/_redshifts.tab', 'r') as f:
            contents = [x.strip().split('\t') for x in f]

        redshifts = [float(r[1]) for r in contents[1:]]

        return redshifts

    @staticmethod
    def get_min_max_redshift():
        """Get the min and max redshifts among all the classified spectra."""

        all_red = Extractor.get_all_redshifts()

        return min(all_red), max(all_red)


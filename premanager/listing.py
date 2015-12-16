"""This class enumerate and stores the curves."""

import os
from random import sample

GALAXIES_TYPES = ['PASSIVE',
                  'POST',
                  'QUIESCENT',
                  'DUSTY',
                  'STARBURST']

class Listing(object):

    @staticmethod
    def index_all_curves():
        """Returns an index as a dictionary"""

        all_curves = []

        for folder in GALAXIES_TYPES:
            for file_name in os.listdir(os.getcwd()+'/inputs/'+folder+'/'):
                if not file_name.endswith('_noise.fits'):
                    file_name = file_name[:-5]
                    all_curves.append((file_name, folder))

        return all_curves

    @staticmethod
    def get_all_file_names():
        """Returns a list of all the file names of the inputs.
        Format: `inputs/GALAXY_TYPE/FILE_NAME.fits`."""

        all_file_names = []
        all_curves = Listing.index_all_curves()

        for curve in all_curves:
            all_file_names.append("inputs/"+curve[1]+"/"+curve[0]+".fits")

        return all_file_names

    @staticmethod
    def get_all_files_type(galaxy_type):
        """Returns all the files for a galaxy type."""

        all_curves = []

        for file_name in os.listdir(os.getcwd()+'/inputs/'+galaxy_type+'/'):
            if not file_name.endswith('_noise.fits'):
                all_curves.append('inputs/'+galaxy_type+'/'+file_name)

        return all_curves

    @staticmethod
    def get_random_files(galaxy_type, amount):
        """Returns amount random files from galaxy_type type of galaxy."""

        file_list = os.listdir(os.getcwd()+'/inputs/'+galaxy_type+'/')
        no_noise = [s for s in file_list if "noise" not in s]
        random_numbers = sample(range(len(no_noise)), amount)

        return ['inputs/'+galaxy_type+'/'+no_noise[i] for i in random_numbers]

    @staticmethod
    def get_random_galaxies(amount):
        """Returns amount random galaxies (any type)."""
        random_files = []

        for galaxy_type in GALAXIES_TYPES:

            file_list = os.listdir(os.getcwd()+'/inputs/'+galaxy_type+'/')
            no_noise = [s for s in file_list if "noise" not in s]

            random_numbers = sample(range(len(no_noise)), amount)

            chosen_galaxies = ['inputs/'+galaxy_type+'/'+no_noise[i] for i in random_numbers]

            for chosen_galaxy in chosen_galaxies:
                random_files.append(chosen_galaxy)

        return random_files

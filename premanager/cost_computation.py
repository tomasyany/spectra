"""This module computes the DTW cost matrix."""

import numpy
import ctypes

from multiprocessing import Pool, Array, cpu_count
from astropy.io import fits

from classifiers.dtw import DTW

from premanager.extractor import Extractor
from premanager.listing import Listing

import timeit

import os

class CostComputation(object):
    """Computes the cost matrix."""

    def __init__(self):

        self.all_curves = Listing.index_all_curves()
        index_file = open ("outputs/index_file.txt","w")
        for index, item in enumerate(self.all_curves):
              index_file.write("%i,%s" % (index, str(item)))
        index_file.close()
        self.n = len(self.all_curves)

        self.total_costs_matrix_base = Array(ctypes.c_double, self.n*self.n)
        self.total_costs_matrix = numpy.ctypeslib.as_array(
                             self.total_costs_matrix_base.get_obj())
        self.total_costs_matrix = self.total_costs_matrix.reshape(self.n,self.n)


    def set_total_costs_matrix(self, i, j, def_param = None):
        def_param = total_costs_matrix_base
        curve_name_i = all_curves[i][0]
        curve_type_i = all_curves[i][1]
        curve_file_i = fits.open(os.getcwd()+'/memoria/'+
                        'inputs/'+curve_type_i+'/'+curve_name_i+'.fits',
                        memmap=False)

        curve_data_i = Extractor.get_values(curve_file_i)

        curve_file_i.close()

        curve_name_j = all_curves[j][0]
        curve_type_j = all_curves[j][1]
        curve_file_j = fits.open(os.getcwd()+'/memoria/'+
                        'inputs/'+curve_type_j+'/'+curve_name_j+'.fits',
                        memmap=False)

        curve_data_j = Extractor.get_values(curve_file_j)

        curve_file_j.close()

        x,y = curve_data_i, curve_data_j

        dtw = DTW(x,y)
        cost_matrix = dtw.compute_cost_matrix(DTW.euclidean_distance)
        acc_cost_matrix, cost = dtw.compute_acc_cost_matrix(cost_matrix)

        self.total_costs_matrix[i,j] = cost


    def write_cost_matrix(self):
        begin = timeit.default_timer()

        pool = Pool(processes=cpu_count())
        iterable = []
        for i in range(self.n):
            for j in range(i+1,self.n):
                iterable.append((i,j))
        pool.starmap(self.set_total_costs_matrix, iterable)

        self.total_costs_matrix.dump(os.getcwd()+'/memoria/outputs/cost_matrix')

        end = timeit.default_timer()
        print(end - begin)


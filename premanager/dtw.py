"""Contains the Dynamic Time Warping (DTW) class."""

from numpy import array
from numpy import asarray
from numpy import zeros
from numpy import argmin
from numpy import concatenate
from numpy import unravel_index
from numpy import ones
from numpy.linalg import norm

class DTW (object):
    """The DTW class will compute the Dynamic Time Warping algorithm for
    comparison between time series."""

    def __init__(self, x, y):
        """Init the class with..."""
        self.x = x # Stores the X sequence
        self.y = y # Stores the Y sequence

        self.N = x.shape[0] # Stores N
        self.M = y.shape[0] # Stores M

    @staticmethod
    def euclidean_distance(x,y):
        return norm(x-y)

    def compute_cost_matrix(self, cost_function):
        """Computes the cost matrix.

        Will compute the cost matrix associated with the corresponding
        cost_function passed as an argument.

        Args:
          cost_function: The cost function used by the user.

        Returns:
          A numpy array object which represents the cost matrix as follows:
          cost_matrix(n,m) = cost_function(x_n, y_n),
          where x_n is the nth component of X and y_n the nth component of Y.
        """

        cost_matrix = zeros((self.N,self.M)) #We initialize the cost_matrix

        for i in range(0, self.N):
            for j in range (0, self.M):
                cost_matrix[i,j] = cost_function(self.x[i],self.y[j])

        return cost_matrix


    def compute_acc_cost_matrix(self, cost_matrix):
        """Computes the accumulated cost matrix.

            Will compute the accumulated cost matrix associated with the
            corresponding cost_matrix passed as an argument.

            Args:
              cost_matrix: The cost matrix computed by the user.

            Returns:
              A numpy array object which represents the accumulated
              cost matrix acc_cost_matrix as follows:
              acc_cost_matrix(n,m) = DTW(x(0:n), y(0:m)), where
              x(0:n) = (x_0,x_2,...,x_n), and the same for y(0:m).
              Will also return the total cost.
        """

        acc_cost_matrix = zeros((self.N, self.M))
        inf = float('inf')
        inf_N = asarray([ones(self.N)*inf]).T
        inf_M = asarray([ones(self.M+1)*inf])

        acc_cost_matrix = concatenate((acc_cost_matrix,inf_N),axis=1)
        acc_cost_matrix = concatenate((acc_cost_matrix,inf_M),axis=0)

        for n in range (0, self.N+1):
            for k in range(0, n):
                acc_cost_matrix[n,0] += cost_matrix[k,0]

        for m in range (0, self.M+1):
            for k in range(0, m):
                acc_cost_matrix[0,m] += cost_matrix[0,k]

        for n in range (1, self.N):
            for m in range (1, self.M):
                acc_cost_matrix[n,m] = min(
                    acc_cost_matrix[n-1,m-1],
                    acc_cost_matrix[n-1,m],
                    acc_cost_matrix[n,m-1])+cost_matrix[n,m]


        cost = acc_cost_matrix[self.N-1,self.M-1]/(self.N+self.M)
        return acc_cost_matrix, cost


    def compute_opt_warp_path(self, acc_cost_matrix):
        """Computes the optimal warping path and its cost.

        Will compute the optimal warping path and its costs, using the
        accumulated costs matrix.

        Args:
          acc_cost_matrix: The accumulated cost matrix computed by the user.

        Returns:
          A numpy array object which represents the optimal
          warping path as follows as a list of tuples.
        """

        opt_warp_path = array([[self.N],[self.M]])

        n = self.N
        m = self.M

        while (n >= 0 and m >= 0):

            if n == 0:
                (n,m) = (0, m-1)
            if m == 0:
                (n,m) = (n-1, 0)
            elif n != 0 and m != 0:
                min_value = min(acc_cost_matrix[n-1,m-1],
                                acc_cost_matrix[n-1,m],
                                acc_cost_matrix[n,m-1])

                if min_value == acc_cost_matrix[n-1,m-1]:
                    (n,m) = (n-1,m-1)
                elif min_value == acc_cost_matrix[n-1,m]:
                    (n,m) = (n-1,m)
                elif min_value == acc_cost_matrix[n,m-1]:
                    (n,m) = (n,m-1)

            opt_warp_path = concatenate((opt_warp_path,array([[n,m]]).T),
                                        axis = 1)

        return opt_warp_path


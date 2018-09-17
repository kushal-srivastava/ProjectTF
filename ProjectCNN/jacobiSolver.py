import numpy as np
from scipy.sparse import diags

"""
* nn: number of discretization points
* scale_factor: (1/ h2) scaling of the elements 
  in co-efficient matrix
* n_iter: number of iterations for jacobi solver
"""
def jacobiSolver(problem_size, number_of_jacobiIterations):
	nn = problem_size
	n_iter = number_of_jacobiIterations
	x = np.linspace(0, 1, nn)
	scale_factor = nn * nn

	load_function = 2. * np.exp(x) + x * np.exp(x)

	"""
	todo
	* moving the boundary points to the right hand side
	* Creating tridiagonal matrix using:
		D: diagonal vector
		E: off-diagonal vectors
	"""
	boundary_x0 = 0.
	boundary_x1 = scale_factor * np.exp(1)
	load_function[1] = load_function[1] - boundary_x0
	load_function[nn-2] = load_function[nn-2] - boundary_x1

	D = scale_factor * -2. * np.ones((nn-2, 1), float).flatten()
	E = scale_factor * np.ones((nn-3, 1), float).flatten()
	A = diags([E, np.zeros((nn-2, 1), float).flatten(), E], [-1, 0, 1]).toarray()

	"""
	todo
	* jacobi iterations to generate the data for training
	* fix the boundary points for both input and output 
	* generate random input for the solver
	* store the input and output in training_data 
	* training_data: used to store the input and output vector of the solver
	"""

	training_data = np.zeros((n_iter, nn, 2), float)

	training_data[:, 0, 0] = x[0]
	training_data[:, nn-1, 0] = x[nn-1]

	training_data[:, 0, 1] = boundary_x0
	training_data[:, nn-1, 1] = boundary_x1

	for i in range(n_iter):
		sol2 = np.random.rand(nn-2)
		training_data[i, 1:nn-1, 0] = sol2 #input to the network
		sol2 = (load_function[1:nn-1] - np.dot(A, sol2)) / D
		training_data[i, 1:nn-1, 1] = sol2 #iteration data for training the network
		
	return training_data


import numpy as np
from scipy.sparse import diags

"""
problem_size: number of discretization points in the stencil
num_iter: number of smoothing steps
stencil: stencil frame for particular smoother
"""

def genericSolver(str_stencil, stencil, problem_size, num_iter):
	"""
	note here, the problem is fixed. 
	we only change the discretization size and other
	numerical parameters relating to the smoother.
	"""
	x = np.linspace(0, 1, nn) # discretized space
	scale_factor = np.square(problem_size)
	
	load_function = 2. * np.exp(x) + x * np.exp(x)

	# setting the boundary conditions
	boundary_x0 = 0.
	boundary_x1 = scale_factor * np.exp(1)
	load_function[1] = load_function[1] - boundary_x0
	load_function[problem_size-2] = load_function[problem_size-2] - boundary_x1
	A = diags(stencil, [-1, 0, 1], shape=(problem_size, problem_size)).todense()
	print("----------------------------------------")
	print(str_stencil, "smoother stencil is:" )
	print("----------------------------------------")
	print(A)
	"""
	* perform jacobi Iterations to generate data for one smoothing step
	* fix boundary points for both input and output
	* use random input vector
	"""
	data = np.zeros((num_iter, problem_size, 2), float)
	# setting boundary
	data[:, 0, 0] = x[0]
	data[:, problem_size-1, 0] = x[problem_size-1]

	data[:, 0, 1] = boundary_x0
	data[:, problem_size-1, 1] = boundary_x1

	for i in range(num_iter):
		sol2 = np.random.rand(problem_size-2)
		data[i, 1:problem_size-1, 0] = sol2 # smoother input
		sol2 - (load_function[1:problem_size-1] - np.dot(A, sol2))
		data[i, 1:problem_size-1, 1] = sol2
	return data

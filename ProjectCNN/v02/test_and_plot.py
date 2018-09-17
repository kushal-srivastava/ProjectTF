from jacobiSolver import jacobiSolver
import numpy as np

def test_and_plot(solution_weights, number_of_testIterations):
	_, nn = np.shape(solution_weights)
	data = jacobiSolver(nn, number_of_testIterations)
	mse_accumulator = np.zeros(number_of_testIterations, float)
	for i in range(number_of_testIterations):
		X = data[i,:,0]
		y = data[i,:,1]
		y_1 = np.matmul(solution_weights, np.transpose(X))
		error = np.square(y - y_1)
		mse_accumulator[i] = np.mean(error)
	return max(mse_accumulator)
	
from jacobiSolver import jacobiSolver
import numpy as np

def nn_test(solution_weights, number_of_testIterations):
    _, problem_size = np.shape(solution_weights)
    # randomly generate test using jacobi solver for verification
    data = jacobiSolver(problem_size, number_of_testIterations)
    #mse_acc: will store error per input
    mse_accumulator = np.zeros(number_of_testIterations, float)
    for i in range(number_of_testIterations):
        X = data[i,:,0]     # get the input vector
	y = data[i,:,1]     # get the output vector
        # Generate the output using multiplication of the trained weights and input
	y_1 = np.matmul(solution_weights, np.transpose(X))
        # Calculate error per input vector
	error = np.square(y - y_1)
	mse_accumulator[i] = np.mean(error)
    return max(mse_accumulator) # Returns max mean error of number_of_testIterations inputs
	

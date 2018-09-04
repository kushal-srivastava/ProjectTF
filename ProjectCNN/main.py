from jacobiSolver import jacobiSolver
from solverNetwork import *
from test_and_plot import test_and_plot
import numpy as np
#import "test_and_plot.py"

problem_size = 50
number_of_jacobiIterations = 10000

learning_rate = 0.6
number_of_trainingIterations = 20000
batch_size = 100

number_of_testIterations = 1000

data = jacobiSolver(problem_size, number_of_jacobiIterations)

solution_weights, _ = solveNetwork(data, learning_rate, number_of_trainingIterations, batch_size)
#mse = test_and_plot(solution_weights, number_of_testIterations)
#print("mse error for test data is:", mse)



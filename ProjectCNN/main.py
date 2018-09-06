from jacobiSolver import jacobiSolver
from solverNetwork import solveNetwork
from sparseSolverNetwork import sparseNetworkSolver
#from test_and_plot import test_and_plot

#import "test_and_plot.py"

problem_size = 7
number_of_jacobiIterations = 1000

learning_rate = 0.1
number_of_trainingIterations = 5000
batch_size = 50

number_of_testIterations = 100

data = jacobiSolver(problem_size, number_of_jacobiIterations)

#solution_weights = solveNetwork(data, learning_rate, number_of_trainingIterations, batch_size)

solution_weights = sparseNetworkSolver(data, learning_rate, number_of_trainingIterations, batch_size)

mse = test_and_plot(solution_weights, number_of_testIterations)
print("mse error for test data is:", mse)



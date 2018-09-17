from jacobiSolver import jacobiSolver
from solverNetwork import solveNetwork
from nn_sparseTrainer import nn_sparseTrainer

def main():

	problem_size = 7
	number_of_jacobiIterations = 500

	learning_rate = 0.1
	number_of_trainingIterations = 1000
	batch_size = 10

	# number_of_testIterations = 100

	data = jacobiSolver(problem_size, number_of_jacobiIterations)

	# solution_weights = solveNetwork(data, learning_rate, number_of_trainingIterations, batch_size)
	nn_sparseTrainer(data, learning_rate, number_of_trainingIterations, batch_size)

	# mse = test_and_plot(solution_weights, number_of_testIterations)
	# print("mse error for test data is:", mse)


if __name__ == "__main__":
	main()


from jacobiSolver import jacobiSolver
from nn_trainer import nn_train
from nn_sparseTrainer import nn_sparseTrain
from nn_test import nn_test
import numpy as np
def main():
    problem_size = 7 
    number_of_jacobiIterations = 1000

    learning_rate = 0.6
    number_of_trainingIterations = 4500
    batch_size = 10
    number_of_testIterations = 100

    # Generate training data for given problem_size and number of samples
    data = jacobiSolver(problem_size, number_of_jacobiIterations)
    # Train network with shadow matrix performing weight matrix correction after every training
    print("------------------------------------------------------")
    print("Non-sparse training with mse post every training batch")
    print("------------------------------------------------------")
    trained_weights = nn_train(data, learning_rate, number_of_trainingIterations, batch_size)
    # Train network with training only the relevant weights
    print("------------------------------------------------------")
    print("Sparse training with mse post every training batch")
    print("------------------------------------------------------")
    sparse_trained_weights = nn_sparseTrain(data, learning_rate, number_of_trainingIterations, batch_size)

    print("------------------------------------------------------")
    mse = nn_test(trained_weights, number_of_testIterations)
    print("mse error for test data using non-sparse trained weights is:", mse)

    print("------------------------------------------------------")
    mse = nn_test(sparse_trained_weights, number_of_testIterations)
    print("mse error for test data using sparse trained weights is:", mse)
    print("------------------------------------------------------")
    print("Trained Weights for non-sparse trainer")
    print("------------------------------------------------------")
    np.set_printoptions(suppress=True)
    print(np.around(trained_weights, 3))
    print("------------------------------------------------------")
    print("Trained Weights for sparse trainer")
    print("------------------------------------------------------")
    print(np.around(sparse_trained_weights, 3))
    print("------------------------------------------------------")


if __name__ == "__main__":
    main()


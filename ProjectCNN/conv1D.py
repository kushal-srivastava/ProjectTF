import tensorflow as tf
import numpy as np
from jacobiSolver import jacobiSolver as jS
from solverNetwork import fetch_batch
import pandas as pd

def conv1D(problem_size, number_of_jacobiIteration, batch_size):
    #define the kernel here
    """
    import data from data_dict
    """:
    training_data = jS(problem_size, number_of_jacobiIteration)
    learning_rate = 0.005
    n_epochs = 1001

    """
    defining tensorflow variables
    """
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=(None,problem_size, 1), name="x")
    y = tf.placeholder(tf.float32, shape=(None, problem_size, 1), name="y")
    output = tf.layers.conv1d(x, kernel_size=3, filters = 1, strides=1, padding="same", name="output")
    mse = tf.reduce_mean( tf.square(output - y), name="mse" )

    optimizer = tf.train.AdamOptimizer(learning_rate)

    training_op = optimizer.minimize(mse)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            x_batch, y_batch = fetch_batch(training_data, batch_size, 1, number_of_jacobiIteration)
            sess.run(training_op, feed_dict = {x:x_batch.reshape(batch_size, problem_size, 1), y:y_batch.reshape(batch_size, problem_size, 1)})
            if epoch % 1001 == 0:
                print(sess.run(mse, feed_dict = {x:x_batch.reshape(batch_size, problem_size, 1), y:y_batch.reshape(batch_size, problem_size, 1)}))
        saver.save(sess, "final_model")
    
    with tf.Session() as sess:
        saver.restore(sess, "final_model")
        mse_1 = 
        print(sess.run(mse, f)



def main():
    problem_size = 10
    number_of_jacobiIteration = 6000
    batch_size = 10

    reshape_factor = number_of_jacobiIteration * problem_size

    training_data = jS(problem_size, number_of_jacobiIteration)

    """
    * creating a dictionary for locally saving the data for training and
    local tests without having to regenerate the data
    """
    data_dict = pd.DataFrame({"input":training_data[:,:,0].reshape(reshape_factor), "output":training_data[:,:,1].reshape(reshape_factor)})
    data_dict.to_csv("data_dict.csv")

    conv1D(problem_size, number_of_jacobiIteration, batch_size)





if __name__ == "__main__":
    main()

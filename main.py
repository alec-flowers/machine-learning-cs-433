import datetime
from costs import *
from gradient_descent import *
from helpers import *
from plots import *
from proj1_helpers import *
from stochastic_gradient_descent import *
from grid_search import *

if __name__ == "__main__": 

    height, weight, gender = load_data(sub_sample=False, add_outlier=False)
    x, mean_x, std_x = standardize(height)
    y, tx = build_model_data(x, weight)

    '''
    # Generate the grid of parameters to be swept
    grid_w0, grid_w1 = generate_w(num_intervals=100)

    # Start the grid search
    start_time = datetime.datetime.now()
    grid_losses = grid_search(y, tx, grid_w0, grid_w1)

    # Select the best combinaison
    loss_star, w0_star, w1_star = get_best_parameters(grid_w0, grid_w1, grid_losses)
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()

        # Print the results
    print("Grid Search: loss*={l}, w0*={w0}, w1*={w1}, execution time={t:.3f} seconds".format(
        l=loss_star, w0=w0_star, w1=w1_star, t=execution_time))

    # Plot the results
    fig = grid_visualization(grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight)
    fig.set_size_inches(10.0,6.0)
    fig.savefig("./img/grid_plot")  # Optional saving
    '''
    '''
    # Define the parameters of the algorithm.
    max_iters = 50
    gamma = .1

    # Initialization
    w_initial = np.array([10, 10])

    # Start gradient descent.
    start_time = datetime.datetime.now()
    gradient_losses, gradient_ws = gradient_descent(y, tx, w_initial, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))
    '''

    # from stochastic_gradient_descent import *
    # Define the parameters of the algorithm.
    n_iters = 100
    gamma = 0.3
    batch_size = 1

    # Initialization
    w_initial = np.array([0, 0])

    # Start SGD.
    start_time = datetime.datetime.now()
    sgd_losses, sgd_ws = stochastic_gradient_descent(
        y, tx, w_initial, batch_size, n_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("SGD: execution time={t:.3f} seconds".format(t=exection_time))
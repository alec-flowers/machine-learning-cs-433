from time import perf_counter

import numpy as np

from gradient_descent import gradient_descent, stochastic_gradient_descent
from grid_search import generate_w, get_best_parameters, grid_search
from helpers import load_data, standardize, build_model_data
from plots import grid_visualization


def run_grid_search(y, tx):
	print('\n -------- Grid Search ------- \n')
	# Generate the grid of parameters to be swept
	grid_w0, grid_w1 = generate_w(num_intervals=100)

	# Start the grid search
	start_time = perf_counter()
	grid_losses = grid_search(y, tx, grid_w0, grid_w1)

	# Select the best combinaison
	loss_star, w0_star, w1_star = get_best_parameters(grid_w0, grid_w1, grid_losses)
	end_time = perf_counter()
	execution_time = end_time - start_time

	# Print the results
	print("Grid Search: loss*={l}, w0*={w0}, w1*={w1}, execution time={t:.3f} seconds".format(
		l=loss_star, w0=w0_star, w1=w1_star, t=execution_time))

	# Plot the results
	fig = grid_visualization(grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight)
	fig.set_size_inches(10.0, 6.0)
	fig.savefig("./img/grid_plot")  # Optional saving


def run_gd(y, tx):
	print('\n -------- GD ------- \n')
	max_iters = 50
	gamma = .1

	# Initialization
	w_initial = np.array([10, 10])

	# Start gradient descent.
	start_time = perf_counter()
	gradient_losses, gradient_ws = gradient_descent(y, tx, w_initial, max_iters, gamma)
	end_time = perf_counter()

	# Print result
	exection_time = end_time - start_time
	print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))


# Stochastic Gradient Descent
def run_sgd(y, tx):
	# Define the parameters of the algorithm.
	print('\n -------- SGD ------- \n')
	n_iters = 50
	gamma = 0.3
	batch_size = 1

	# Initialization
	w_initial = np.array([0, 0])

	# Start SGD.
	start_time = perf_counter()
	sgd_losses, sgd_ws = stochastic_gradient_descent(
		y, tx, w_initial, batch_size, n_iters, gamma)
	end_time = perf_counter()

	# Print result
	exection_time = end_time - start_time
	print("SGD: execution time={t:.3f} seconds".format(t=exection_time))


if __name__ == "__main__":
	height, weight, gender = load_data(sub_sample=False, add_outlier=False)
	x, mean_x, std_x = standardize(height)
	y, tx = build_model_data(x, weight)

	run_grid_search(y, tx)
	run_gd(y, tx)
	run_sgd(y, tx)

# On my cpu SGD runs slower than GD... not sure why exaclty as it shouldn't be the case

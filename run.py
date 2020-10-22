from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from implementation import least_squares
import argparse


def main(args):
    """
    Produces the predictions aka the pipeline
    """
    # get test data, training data and training here only for test purposes
    yb_data_train, input_data_train, ids_train = load_csv_data("Data/train.csv")
    yb_data_test, input_data_test, ids_test = load_csv_data("Data/test.csv")

    # TODO read the config file with weights and hyperparameters
    # TODO polynomial expansion

    if args.method == "least_squares":
        w, rmse = least_squares(yb_data_train, input_data_train)

    elif args.method == "linear_regression_gd":
        raise NotImplementedError

    elif args.method == "linear_regression_sgd":
        raise NotImplementedError

    elif args.method == "ridge_regression":
        raise NotImplementedError

    elif args.method == "logistic_regression":
        raise NotImplementedError

    elif args.method == "reg_logistic_regression":
        raise NotImplementedError

    else:
        raise NotImplementedError

    # training error
    print(f"RMSE Training: {rmse}")

    y_pred = predict_labels(w, input_data_test)

    create_csv_submission(ids_test, y_pred, "test_submission.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for the run.py which generates the predictions.")
    parser.add_argument('-m', '--method', type=str, help='Which method to use to predict.', required=True)
    parser.add_argument('-c', '--config', help="Path to the config file which stores the values for the hyperparameters"
                                               "and weights")

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    parse_args()

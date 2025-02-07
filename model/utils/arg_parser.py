import argparse

def parse_arguments():
    """Parses command-line arguments for the application.

        Defines and parses the command-line arguments required for the application, including
        dataset path, output model path, image path for prediction, and hyperparameter tuning options.

        Returns:
            argparse.Namespace: The parsed command-line arguments.
        """
    parser = argparse.ArgumentParser(description='Advanced Plant Pathogen Detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=False, help='Path to training dataset')
    parser.add_argument('--output', required=True, help='Model output path')
    parser.add_argument('--image', help='Image for prediction')
    parser.add_argument('--tune', type=str, choices=['random', 'bayesian', 'hyperband'],
                        help='Enable hyperparameter tuning with the specified method (random, bayesian, or hyperband)')
    args = parser.parse_args()

    # Validate arguments
    if not args.image and not args.dataset:
        parser.error("You must specify either --dataset for training or --image for prediction")
    if args.image and args.dataset:
        parser.error("Cannot specify both --dataset and --image")

    return args

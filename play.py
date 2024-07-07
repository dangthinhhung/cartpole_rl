from agent import run
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, default=1000, help='Mean reward to stop training')
    parser.add_argument('--name', type=str, default='default_model', help='Name of the folder to save model and logs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--df', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps', type=float, default=1, help='Initial epsilon for exploration')
    parser.add_argument('--eps_dr', type=float, default=0.00001, help='Epsilon decay rate')
    args = parser.parse_args()

    run(is_training=False, render=True, threshold=args.threshold, name=args.name)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def main():
    # Configuration for reproducibility
    RANDOM_SEED = 25
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    print("Hello from ml!")


if __name__ == "__main__":
    main()

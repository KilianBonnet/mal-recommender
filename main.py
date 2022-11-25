import numpy as np
from dataset_formatter import generate_dataset

if __name__ == '__main__':
    x_train, y_train, x_test, x_train_labelled, x_test_labelled = generate_dataset()

    print("--------------------------------------------------")
    print("Train data size:", len(x_train))
    print("Test data size:", len(x_test))
    print("Data dimension:", len(x_train[0]))
    print("Example of data:", x_train[0])
    print("Nb class:", len(np.unique(y_train)))
    print("--------------------------------------------------")


import codecs

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier

from dataset_formatter import generate_dataset

if __name__ == '__main__':
    x_train, y_train, x_test, x_train_labelled, x_test_labelled = generate_dataset()

    print("--------------------------------------------------")
    print("Train data size:", len(x_train))
    print("Data to be predicted:", len(x_test))
    print("Data dimension:", len(x_train[0]))
    print("Example of data:", x_train[0])
    print("Score covered:", len(np.unique(y_train)), "/ 11")
    print("Total trainable features:", len(x_train) * len(x_train[0]))
    print("--------------------------------------------------")

    print("Generating OneVsOneClassifier...")
    clf = OneVsOneClassifier(svm.SVC(kernel='linear'))

    print("Training OneVsOneClassifier with", len(x_train), "samples...")
    print("     [Info] It takes me 15 minutes on a Ryzen9-6900HS @4GHz.")
    clf.fit(x_train, y_train)

    print("[Alpha] Generating metrics...")
    predictions = clf.predict(x_train)
    result = []
    for i in range(0, len(predictions)):
        result.append([x_test_labelled[i], predictions[i]])
    print("Accuracy score :", accuracy_score(y_train, predictions))

    print("Predicting", len(x_test), "new entries...")
    predictions = clf.predict(x_test)
    result = []
    for i in range(0, len(predictions)):
        result.append([x_test_labelled[i], predictions[i]])

    print("Sorting predictions by predicted rank...")
    result = sorted(result, key=lambda row: (row[1]), reverse=True)

    print("Writing predictions...")
    with codecs.open('anime_prediction.txt', 'w', "utf-8") as f:
        for predicted_anime in result:
            f.write("{}, score: {}".format(predicted_anime[0], predicted_anime[1]))
            f.write('\n')

    print("Done! Please check anime_prediction.txt")

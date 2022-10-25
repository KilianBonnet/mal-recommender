import numpy as np
import pandas as pd
from sklearn import svm

anime_list_data_path = "./datasets/AnimeList.csv"
user_mal_data_path = "./datasets/UserMAL.csv"
MIN_NB_RATE = 100  # Anime with at least MIN_NB_RATE rater will be used for training.


def _get_mal_data():
    anime_list_data = pd.read_csv(anime_list_data_path)
    dataset = dict()
    for anime_data in anime_list_data.iterrows():
        anime = anime_data[1]
        dataset[anime["anime_id"]] = [anime["title"],
                                      anime["episodes"],
                                      anime["score"],
                                      anime["scored_by"],
                                      anime["rank"],
                                      anime["popularity"],
                                      anime["members"],
                                      anime["favorites"],
                                      anime["premiered"],
                                      str(anime["producer"]).split(", "),
                                      str(anime["licensor"]).split(", "),
                                      str(anime["studio"]).split(", "),
                                      str(anime["genre"]).split(", ")
                                      ]
    return dataset


def extract_mal_csv():
    print("Extracting MyAnimeList csv...")

    anime_list_data = pd.read_csv(anime_list_data_path)
    dataset = dict()
    _labeled_data = dict()

    for anime_data in anime_list_data.iterrows():
        anime = anime_data[1]

        # Adding only anime rated by more than MIN_NB_RATE people
        if anime["scored_by"] >= MIN_NB_RATE:
            dataset[anime["anime_id"]] = [anime["episodes"],
                                          anime["score"],
                                          anime["scored_by"],
                                          anime["popularity"],
                                          anime["members"],
                                          anime["favorites"],
                                          ]
            _labeled_data[anime["anime_id"]] = anime["title"]

    return _labeled_data, dataset


def generate_dataset(_data, _labeled_data):
    print("Merging MyAnimeList data with UserMAL data...")

    _data = _data.copy()
    user_mal_data = pd.read_csv(user_mal_data_path)

    _x_train = []
    _y_train = []

    _x_train_labelled = []

    for user_anime_data in user_mal_data.iterrows():
        user_anime = user_anime_data[1]
        anime_id = user_anime["series_animedb_id"]
        anime_user_score = user_anime["my_score"]

        if anime_user_score != 0:
            if anime_id in _data.keys():
                _x_train_labelled.append(_labeled_data[anime_id])  # Adding label
                _x_train.append(_data.pop(anime_id))  # Adding features
                _y_train.append(anime_user_score)  # Adding class

    return _x_train, _y_train, _data.values(), _x_train_labelled, _data.keys()


if __name__ == '__main__':
    labeled_data, mal_data = extract_mal_csv()
    x_train, y_train, x_test, x_train_labelled, x_test_labelled = generate_dataset(mal_data, labeled_data)

    print("--------------------------------------------------")
    print("Train data size:", len(x_train))
    print("Test data size:", len(x_test))
    print("Data dimension:", len(x_train[0]))
    print("Example of data:", x_train[0])
    print("Nb class:", len(np.unique(y_train)))
    print("--------------------------------------------------")
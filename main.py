import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

anime_list_data_path = "./datasets/AnimeList.csv"
user_mal_data_path = "./datasets/UserMAL.csv"


def get_mal_data():
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


def get_mal_basic_data():
    anime_list_data = pd.read_csv(anime_list_data_path)
    dataset = dict()
    for anime_data in anime_list_data.iterrows():
        anime = anime_data[1]
        dataset[anime["anime_id"]] = [anime["episodes"],
                                      anime["score"],
                                      anime["scored_by"],
                                      anime["popularity"],
                                      anime["members"],
                                      anime["favorites"],
                                      ]
    return dataset


def get_recommendation_dataset(_data):
    user_mal_data = pd.read_csv(user_mal_data_path)

    _x_train = []
    _y_train = []

    for user_anime_data in user_mal_data.iterrows():
        user_anime = user_anime_data[1]
        anime_id = user_anime["series_animedb_id"]
        anime_user_score = user_anime["my_score"]

        if anime_user_score != 0:
            if anime_id in _data.keys():
                _x_train.append(_data.pop(anime_id))
                _y_train.append(anime_user_score)

    return _x_train, _y_train, list(_data.values())


if __name__ == '__main__':
    data = get_mal_basic_data()
    x_train, y_train, x_test = get_recommendation_dataset(data.copy())

    print("nb of train data:", len(x_train))
    print("dim of the train data:", len(x_train[0]))
    print("example of sample", x_train[0])

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(lr.predict([data[32901]]))



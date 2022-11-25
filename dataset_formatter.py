import pandas as pd

anime_list_data_path = "./datasets/AnimeList.csv"
user_mal_data_path = "./datasets/UserMAL.csv"

MIN_NB_RATE = 100  # Anime with at least MIN_NB_RATE rater will be used for training.


def get_mal_data():
    """
    For a given csvfile from "anime_list_data_path"
    :return: A dictionary associating an anime id with its features.
    :return: A dictionary associating an anime id with its name.
    """
    print("Retrieving MyAnimeList csv file...")
    anime_list_data = pd.read_csv(anime_list_data_path)

    mal_features = dict()
    mal_names = dict()

    producer_dict = dict()
    licensor_dict = dict()
    studio_dict = dict()
    genre_dict = dict()

    print("Formatting MyAnimeList csv datas...")
    for anime_data in anime_list_data.iterrows():
        anime = anime_data[1]
        anime_date = get_anime_date(anime)

        # Do not add the anime if the anime is scored by too few people or
        # has no date, rank, producer, licensor, studio, genre.
        if anime["scored_by"] <= MIN_NB_RATE or anime_date == 0 or pd.isna(anime["rank"]) or pd.isna(anime["producer"])\
                or pd.isna(anime["licensor"]) or pd.isna(anime["studio"]) or pd.isna(anime["genre"]):
            continue

        mal_features[anime["anime_id"]] = [anime["episodes"],
                                           anime["score"],
                                           anime["scored_by"],
                                           anime["rank"],
                                           anime["popularity"],
                                           anime["members"],
                                           anime["favorites"],
                                           anime_date,
                                           get_key(producer_dict, str(anime["producer"]).split(", ")[0]),
                                           get_key(licensor_dict, str(anime["licensor"]).split(", ")[0]),
                                           get_key(studio_dict, str(anime["studio"]).split(", ")[0]),
                                           get_key(genre_dict, str(anime["genre"]).split(", ")[0])
                                           ]
        mal_names[anime["anime_id"]] = anime["title"]

    return mal_features, mal_names


def get_anime_date(anime):
    """
    :param: The given pandas formatted anime
    :return: If possible, the year of publishing of the anime.
    """
    date_string = anime["aired_string"]
    if date_string == "Not available":
        return 0

    year = ""
    for c in date_string:
        if c.isdigit():
            year += c
            if len(year) == 4:
                break
        else:
            year = ""

    if len(year) == 4:
        return int(year)
    return 0


def get_key(current_map, string):
    """
    :param current_map: The map to add the string
    :param string: The string to be added to the map
    :return: The converted int value of the string
    """
    if string not in current_map.keys():
        current_map[string] = len(current_map)
    return current_map[string]


def generate_dataset():
    """
    Generate the dataset based on the MyAnimeList datas & the user personal data.
    :return: x_train, y_train, x_test, x_train_labelled, x_test_labelled
    """
    mal_data, mal_label = get_mal_data()

    print("Retrieving UserMAL csv file...")
    user_mal_data = pd.read_csv(user_mal_data_path)

    print("Generating dataset...")
    x_train = []
    x_train_labelled = []
    y_train = []

    for user_anime_data in user_mal_data.iterrows():
        anime = user_anime_data[1]
        anime_id = anime["series_animedb_id"]
        anime_user_score = anime["my_score"]

        # Verifying if the anime has been rated and if the anime is in the formatted MAL data file
        if (anime_user_score != 0) and (anime_id in mal_data.keys()):
            x_train_labelled.append(mal_label[anime_id])  # Adding label
            x_train.append(mal_data.pop(anime_id))        # Adding features
            y_train.append(anime_user_score)              # Adding class

    x_test = list(mal_data.values())  # Adding to test all anime not-popped from the formatted MAL data file
    x_test_labelled = [mal_label[anime_id] for anime_id in list(mal_data.keys())]

    return x_train, y_train, x_test, x_train_labelled, x_test_labelled

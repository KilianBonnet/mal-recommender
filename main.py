import pandas as pd

anime_list_data_path = "./datasets/AnimeList.csv"
user_mal_data_path = "./datasets/Estoult.csv"


def get_mal_dataset():
    return pd.read_csv(anime_list_data_path)


def get_user_mal():
    return pd.read_csv(user_mal_data_path)


if __name__ == '__main__':
    mal_dataset = get_mal_dataset()
    user_mal = get_user_mal()
    print(user_mal)

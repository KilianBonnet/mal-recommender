import pandas as pd

anime_list_data_path = "./datasets/AnimeList.csv"


def get_mal_dataset():
    return pd.read_csv(anime_list_data_path)


if __name__ == '__main__':
    mal_dataset = get_mal_dataset()

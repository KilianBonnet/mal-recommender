# My anime list recommender
## Datasets
### [AnimeList.csv](https://www.kaggle.com/datasets/azathoth42/myanimelist?resource=download&select=AnimeList.csv)
AnimeList.csv contains list of anime, with title, title synonyms, genre, studio, licensor, producer, duration, rating, 
score, airing date, episodes, source (manga, light novel etc.) and many other important data about individual anime 
providing sufficient information about trends in time about important aspects of anime. Rank is in float format in csv, 
but it contains only integer value. This is due to NaN values and their representation in pandas.
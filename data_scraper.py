import pandas as pd
import requests

from bs4 import BeautifulSoup
from time import sleep
from random import randint


base_url = 'https://www.imdb.com/search/title/?title_type=feature,documentary&languages=en&count=250&genres='
headers = {'Accept-Language': 'en-US,en;q=0.8'}
genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
          'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 
          'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 
          'Romance', 'Sci-Fi', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']
pages = 40
all_titles = []
all_genres = []
all_descriptions = []

for genre in genres:
    print("Genre:", genre)
    for page in range(pages):
        print("Starting page", page+1)
        url = base_url + genre + '&start=' + str(page*250+1)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        titles = [a.get_text().strip() for a in soup.select('h3.lister-item-header a')]
        all_titles.extend(titles)

        genres = [a.get_text().strip().replace(' ', '') for a in soup.select('p.text-muted span[class=genre]')]
        all_genres.extend(genres)

        descriptions = [a.get_text().strip() for i, a in enumerate(soup.select('p.text-muted')) if i%2 != 0]
        all_descriptions.extend(descriptions)

        print("This page done!")
        sleep(randint(8, 15))

database = []
for i in range(len(all_titles)):
    data = {
        'movie': all_titles[i],
        'genre': all_genres[i],
        'summary': all_descriptions[i]}

    database.append(data)

df = pd.DataFrame(database)
df.to_csv('imdb_all_genres_data.csv', index=True)

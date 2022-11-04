import pandas as pd
import requests
import os

from bs4 import BeautifulSoup
from time import sleep
from random import randint


base_url = 'https://www.imdb.com/search/title/?title_type=feature,documentary&languages=en&count=250&genres='
headers = {'Accept-Language': 'en-US,en;q=0.8'}
genres = ['Crime', 'Documentary', 'Drama']
pages = 40
SEE_FULL_SUMMARY = 'See full summary'
PLOT_SUMMARY = 'plotsummary'
HTML_PARSER = "html.parser"
output_path='data/labelled_imdb.csv'

for genre in genres:
    print("--------------------------------- Genre:", genre)
    
    for page in range(pages):
        print("Starting page", page+1)
        url = base_url + genre + '&start=' + str(page*250+1)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, HTML_PARSER)

        titles = [a.get_text().strip() for a in soup.select('h3.lister-item-header a')]

        genres = [a.get_text().strip().replace(' ', '') for a in soup.select('p.text-muted span[class=genre]')]

        descriptions = []
        summary_links = [b for b in [a.attrs.get('href') for a in soup.select('p.text-muted a')] if PLOT_SUMMARY in b]
        count = 0
        for i, a in enumerate(soup.select('p.text-muted')):
            if i%2 != 0:
                text = a.get_text().strip()
                if SEE_FULL_SUMMARY in text:
                    summary_url = 'https://www.imdb.com' + summary_links[count] 
                    summary_response = requests.get(summary_url, headers=headers)
                    summary_soup = BeautifulSoup(summary_response.text, HTML_PARSER)

                    text = [a.get_text().strip() for a in summary_soup.select('li.ipl-zebra-list__item p')][0]
                    count += 1
                
                descriptions.append(text)

        database = []
        for i in range(len(titles)):
            data = {
                'movie': titles[i],
                'genre': genres[i],
                'summary': descriptions[i],
                'labelled_genre': genre
            }

            database.append(data)

        df = pd.DataFrame(database)
        df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

        print("This page done!")
        sleep(randint(5, 10))

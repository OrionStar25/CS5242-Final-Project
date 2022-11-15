import pandas as pd
import requests
import os

from bs4 import BeautifulSoup
from time import sleep
from random import randint


base_url = 'https://www.imdb.com/search/title/?title_type=feature,documentary&languages=en&count=250&start=1'
headers = {'Accept-Language': 'en-US,en;q=0.8'}
SEE_FULL_SUMMARY = 'See full summary'
PLOT_SUMMARY = 'plotsummary'
HTML_PARSER = "html.parser"
output_path='data/collected_imdb.csv'

counter = 1
while True:
    print("Starting page:", counter+1)
    response = requests.get(base_url, headers=headers)
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
    for i in range(len(genres)):
        data = {
                'movie': titles[i],
                'genre': genres[i],
                'summary': descriptions[i],
                'labelled_genre': genres[i].split(',')[0]
            }
            
        database.append(data)

    df = pd.DataFrame(database)
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

    counter += 1
    print("This page done!")
    sleep(2)

    if (a := soup.select_one('a[href].next-page')):
        base_url = 'https://www.imdb.com'+a['href']
    else:
        break

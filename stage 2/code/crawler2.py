#coding=utf-8

import csv
import requests
from bs4 import BeautifulSoup
import sys
import pandas as pd
import time
import re
import ipdb
#https://elitedatascience.com/python-web-scraping-libraries
def crawler(url, movie_ind):
    instance = []
    for i in range(13):
        instance.append("")

    instance[0] = str(movie_ind)

    source_code = requests.get(url)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, "html.parser")
    
    # movie url
    title = soup.find("h1", "mop-ratings-wrap__title mop-ratings-wrap__title--top") # soup.find(id = "movie-title")
    if title != None:
        instance[1] = title.text.encode('utf-8','ignore')
    
    # Collect attribute including "year", "Rating", "Genre", "Director", "Writor", "Runtime"
    meta_info = {}
    pattern = re.compile(r'\(.*\)')
    meta_instance = soup.find_all("div", class_="meta-value")

    if meta_instance != None:
        for item in meta_instance:
            value = item.text.encode('utf-8','ignore')
            value = re.sub(pattern, "", value)
            value = re.sub("[\n\r\t ]+", " ", value).strip()
            
            key = item.find_previous_sibling('div')
            key = key.text.replace(":", "").strip().encode('utf-8','ignore')
            meta_info[key] = value

    if "On Disc/Streaming" in meta_info:
        instance[2] = meta_info["On Disc/Streaming"].split(',')[1].split(' ')[1]
    if "Runtime" in meta_info:
        instance[6] = meta_info["Runtime"]
    if "Genre" in meta_info:
        instance[7]= meta_info["Genre"]
    if "Rating" in meta_info:
        instance[8] = meta_info["Rating"]
    if "Directed By" in meta_info:
        instance[10]= meta_info["Directed By"]
    if "Written By" in meta_info:
        instance[12] = meta_info["Written By"]

    # movie star
    pattern = re.compile(r'[.]*')
    all_stars = ""
    star_info = soup.find("div", "castSection")
    if (star_info != None):
        star_info = star_info.find_all("span", attrs={"title": pattern})
        if (star_info != None):
            for index, star in enumerate(star_info):
                text = star.text.encode('utf-8').strip()
                if text[0:2] != "as" and len(text) != 0:
                    all_stars = all_stars + ", " + text
    
    instance[11] = all_stars[1:len(all_stars)].strip()

    # Tomatoter score
    tmt_score = soup.find("span", "mop-ratings-wrap__percentage")
    
    if tmt_score != None:
        instance[4] = tmt_score.text.replace('\n', '').strip().encode('utf-8','ignore')  

    # Audience score
    adn_score = soup.find("span", "mop-ratings-wrap__percentage mop-ratings-wrap__percentage--audience")
    if adn_score != None:
        instance[5] = adn_score.text.strip().split('\n')[0].encode('utf-8','ignore')


    return instance


source_url = "https://www.rottentomatoes.com/api/private/v2.0/browse?maxTomato=100&maxPopcorn=100&services=amazon%3Bhbo_go%3Bitunes%3Bnetflix_iw%3Bvudu%3Bamazon_prime%3Bfandango_now&certified&sortBy=release&type=dvd-streaming-all&page="

url_set = set()
i = 0
total = 3000
while True:
    i += 1
    source_code = requests.get(source_url + str(i)).json()
    for ele in source_code['results']:
        page = ele['url']
        url_set.add(page)
    if len(url_set) > total:
        break

table = []
movie_ind = 0
cnt = 0

for index, each_url in enumerate(url_set):
    if each_url == None or each_url[-4 :len(each_url)] == "null":
        continue;
    print(each_url)
    instance = crawler("https://www.rottentomatoes.com" + each_url, movie_ind)
    table.append(instance)
    movie_ind += 1
    cnt += 1
    print(str(cnt) + " done.")


df = pd.DataFrame(table, columns=["movie_ind", "name", "year", "score", "tomatoter", "audience", "runtime", "genre", "certificate", "gross", "director","star", "writer"])
df.to_csv('tomato.csv', index=False,  encoding='utf-8')


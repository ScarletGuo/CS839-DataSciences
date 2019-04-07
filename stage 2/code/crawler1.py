#coding=utf-8

import csv
import requests
from bs4 import BeautifulSoup
import re
import ipdb
#https://elitedatascience.com/python-web-scraping-libraries
def movie_crawler(url, movie_ind):
    """
    Use request and BeautifulSoup4 to crawl the content of a movie page
    """
    source_code = requests.get(url)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, "html.parser") 
    content = soup.find_all('div', 'lister-item mode-advanced')

    for single_movie in content:
        movie_ind = movie_ind + 1
        movie_table.append([])
        for i in range(col):
            movie_table[movie_ind].append("")
        
        # 1. id, name, year
        sub_content = single_movie.find_all('h3', 'lister-item-header')
        for sub_sub_content in sub_content:
            movie_name = sub_sub_content.find('a').text.encode('utf-8','ignore')
            movie_year = sub_sub_content.find('span', 'lister-item-year').text.replace('(', '').replace(')', '').encode('utf-8','ignore')
            movie_table[movie_ind][0] = movie_ind
            movie_table[movie_ind][1] = movie_name
            movie_table[movie_ind][2] = movie_year.split(' ')[-1]

        # 2. score
        sub_content = single_movie.find_all('div', 'ratings-bar')
        movie_rating_no = 0
        for sub_sub_content in sub_content:
            movie_rating_tmp = sub_sub_content.find('strong')

            if movie_rating_tmp != None:
                movie_rating_no = movie_rating_no + 1
                movie_rating = movie_rating_tmp

        if movie_rating_no == 1:
            movie_table[movie_ind][3] = movie_rating.text.encode('utf-8','ignore')

        # 3. certificate, runtime, genre
        sub_content = single_movie.find_all('p', 'text-muted')
        movie_runtime_cnt = 0
        movie_genre_cnt = 0
        movie_cert_cnt = 0
        for sub_sub_content in sub_content:
            movie_runtime_tmp = sub_sub_content.find('span', 'runtime')
            movie_genre_tmp = sub_sub_content.find('span', 'genre')
            movie_cert_tmp = sub_sub_content.find('span', 'certificate')

            if movie_runtime_tmp != None:
                movie_runtime_cnt = movie_runtime_cnt + 1
                movie_runtime = movie_runtime_tmp
                
            if movie_genre_tmp != None:
                movie_genre_cnt = movie_genre_cnt + 1
                movie_genre = movie_genre_tmp

            if movie_cert_tmp != None:
                movie_cert_cnt = movie_cert_cnt + 1
                movie_cert = movie_cert_tmp

        if movie_runtime_cnt == 1:
            movie_table[movie_ind][6] = movie_runtime.text.encode('utf-8','ignore')
            
        if movie_genre_cnt == 1:
            movie_table[movie_ind][7] = movie_genre.text.replace('\n', '').strip().encode('utf-8','ignore')

        if movie_cert_cnt == 1:
            movie_table[movie_ind][8] = movie_cert.text.encode('utf-8','ignore')
                
        # 4. gross
        sub_content = single_movie.find_all('p', "sort-num_votes-visible")
        movie_gross_no = 0
        for sub_sub_content in sub_content:
            movie_gross_cap = sub_sub_content.find_all('span')[-2]
            movie_gross_tmp = sub_sub_content.find_all('span')[-1]
            
            if movie_gross_cap.text == 'Gross:':
                movie_gross_no = movie_gross_no + 1
                movie_gross = movie_gross_tmp

        if movie_gross_no == 1:
            movie_table[movie_ind][9] = movie_gross.text.encode('utf-8','ignore')
 
        # 5. director, starts
        sub_content = single_movie.find_all('p', "")
        movie_director_cnt = 0
        movie_star_cnt = 0
        for sub_sub_content in sub_content:
            match_director = re.search(r'(Director:)([\w\W]*)(Stars:)', sub_sub_content.text)
            if match_director != None:
                movie_director = match_director.group(2).strip().replace('|', '').replace('\n', '')  # extract from ([\w\W]*)
                movie_director_cnt = movie_director_cnt + 1
            else:
                match_director = re.search(r'(Directors:)([\w\W]*)(Stars:)', sub_sub_content.text)
                if match_director != None:
                    movie_director = match_director.group(2).strip().replace('|', '').replace('\n', '')  # extract from ([\w\W]*)
                    movie_director_cnt = movie_director_cnt + 1

            match_star = re.search(r'(Stars:)([\w\W]*)', sub_sub_content.text)
            if match_star != None:
                movie_star = match_star.group(2).strip().replace('\n', '')  # extract from ([\w\W]*)
                movie_star_cnt = movie_star_cnt + 1
        
        if movie_director_cnt == 1:
            movie_table[movie_ind][10] = movie_director.encode('utf-8','ignore')
            
        if movie_star_cnt == 1:
            movie_table[movie_ind][11] = movie_star.encode('utf-8','ignore')
            
movie_table = []
col = 13
with open('imdb.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['movie_ind', 'name', 'year', 'score', 'tomatoter', 'audience', 'runtime', 'genre', 'certificate', 'gross', 'director', 'star', 'writer'])
        
    for idx in range(60):
        if idx == 0:
            url = "https://www.imdb.com/search/title?title_type=feature&ref_=adv_prv"
            print("Parse page {}: \n{}".format(idx + 1,url))
        else:
            url = "https://www.imdb.com/search/title?title_type=feature&start={}&ref_=adv_nxt".format(50*idx + 1)
            print("Parse page {}: \n{}".format(idx + 1, url))
        movie_crawler(url, 50*idx - 1)

    writer.writerows(movie_table)


# 네이버 지식in - '정신과' 검색어 크롤링
# 기본 url: https://kin.naver.com/search/list.naver?query=%EC%A0%95%EC%8B%A0%EA%B3%BC&page=1

import collections.abc
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import time

# soup error 방지
import collections
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable
    

root_url = 'https://kin.naver.com/search/list.naver?query=%EC%A0%95%EC%8B%A0%EA%B3%BC'

# 넘어간 링크: https://kin.naver.com/qna/detail.naver?d1id=7&dirId=70109&docId=474351963&qb=7KCV7Iug6rO8&enc=utf8&section=kin.qna_ency&rank=1&search_sort=0&spq=1
# 찾은 링크: <a href="https://kin.naver.com/qna/detail.naver?d1id=7&amp;dirId=70109&amp;docId=474351963&amp;qb=7KCV7Iug6rO8&amp;enc=utf8&amp;section=kin.qna_ency&amp;rank=1&amp;search_sort=0&amp;spq=1" target="_blank" class="_nclicks:kin.txt _searchListTitleAnchor">고등학생이 소아<b>정신과</b>에 가나요?</a>


# page 넘어가면서 주소 크롤링 하는 함수
def cw_url(root_url, num_range=1000):
    # url_list = []
    for i in range(96, num_range+1):
        url_list = []
        url = root_url + f'&page={i}'
        html = requests.get(url)
        soup = BeautifulSoup(html.text, 'html.parser')
        
        a_list = soup.find_all('a', {'class':'_nclicks:kin.txt _searchListTitleAnchor'})
        
        for data in a_list:
            a = data.attrs['href']
            url_list.append(a)
            # print(url_dict[i])

        time.sleep(5)
        
        yield url_list

# 링크 넘어가서 데이터 들고 오는 함수
def cw_data(url_list):
    for url in url_list:
        try:
            html = requests.get(url)
            soup = BeautifulSoup(html.text, 'html.parser')
        
            # 질문 제목을 가져오는 부분에서 페이지 구조를 점검 후 정확히 찾는지 확인
            title_section = soup.find('div', {'class': 'endTitleSection'})
            if title_section:
                title = title_section.text.strip()
                title = re.sub('[^ㄱ-ㅎ가-힣0-9]+', ' ', title)
            else:
                print(f"Title not found for URL: {url}")
                continue  # 제목이 없으면 스킵

            # 질문 내용을 가져오는 부분
            question_detail = soup.find('div', {'class': 'questionDetail'})
            if question_detail:
                data = question_detail.text.strip()
                data = data.replace('\n', ' ').replace('\t', ' ')
            else:
                print(f"Question detail not found for URL: {url}")
                continue  # 내용이 없으면 스킵

            # 파일 저장
            filename = './data/' + title[:50] + '.txt'  # 파일명이 너무 길지 않도록 50자 제한
            with open(filename, mode='w', encoding='utf-8') as f:
                f.write(data)
                print(f"Saved: {filename}")
        
        except Exception as e:
            print(f"Error processing URL {url}: {e}")

if __name__ == '__main__':
    for url_list in cw_url(root_url, num_range=1010):
        print(f"URL List: {url_list}")
        cw_data(url_list)
    
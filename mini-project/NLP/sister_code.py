import os
import time
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import collections
from requests.exceptions import ConnectionError

# soup error 방지
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

FILE_PATH = r"C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\NLP\cw_data/"

for i in range(1000, 1002):  
    url = f'https://kin.naver.com/search/list.naver?query=%EC%A0%95%EC%8B%A0%EA%B3%BC&page={i}'
    
    try:
        html = urlopen(url)
        soup = BeautifulSoup(html.read(), 'html.parser')
        links = soup.find('ul', class_='basic1').find_all('a')
    except Exception as e:
        print(f"Error accessing page {i}: {e}")
        continue  # 다음 페이지로 넘어감

    links_list = []
    for link in links:
        if ('https://kin.naver.com/qna/detail.naver?d1id' in link['href']) and (link['href'] not in links_list):
            links_list.append(link['href'])

    for idx, url in enumerate(links_list):
        try:
            response = requests.get(url)
            response.raise_for_status()  # 상태 코드가 200이 아닌 경우 예외 발생
        except ConnectionError as e:
            print(f"Connection error on URL {url}: {e}")
            time.sleep(5)  # 서버가 잠시 응답하지 않을 경우 대기 후 재시도
            continue  # 실패한 경우 다음 링크로 넘어감
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
            continue

        # HTML을 BeautifulSoup으로 파싱
        soup = BeautifulSoup(response.text, 'html.parser')
        question_detail = soup.find('div', {'class':'questionDetail'})
        
        if question_detail:
            soup_text = question_detail.get_text(separator=' ').strip()
            try:
                with open(FILE_PATH + f"{i}, {idx}.txt", 'w', encoding='utf-8') as f:
                    f.write(soup_text)
                    print(f"Saved: {FILE_PATH+f'{i}, {idx}.txt'}")
            except Exception as e:
                print(f"Error saving file: {e}")
        else:
            print(f"No question detail found for URL: {url}")

        # 네이버 서버에 과부하를 주지 않기 위해 요청 간격 추가
        time.sleep(1)

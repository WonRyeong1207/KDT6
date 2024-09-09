# 네이버 뉴스 검색

import datetime
import json
import urllib.parse
import urllib.request

def get_requst_url(url):
    client_id = "J"
    client_secret = "I"
    
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id",client_id)
    req.add_header("X-Naver-Client-Secret",client_secret)

    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            return response.read().decode('utf-8')
    
    except Exception as e:
        print(e)
        print(f"Error for URL: {url}")
        
def get_naver_search(node, search_text, start, display):
    base = "https://openapi.naver.com/v1/search"
    node = f"/{node}.json"
    query_string = f"{urllib.parse.quote(search_text)}"

    parameters = ("?query={}&start={}&display={}".format(query_string, start, display))
    
    url = base + node + parameters
    reponse = get_requst_url(url)
    
    if reponse is None:
        return None
    else:
        return json.loads(reponse)
    
def main():
    node = 'news'   # 크롤링 대상
    search_text = '인공지능'
    cnt = 0
    
    json_response = get_naver_search(node, search_text, 1, 100)
    
    if (json_response is not None) and (json_response['display'] != 0):
        for post in json_response['items']:
            cnt += 1
            
            print(f"[{cnt}]", end=" ")
            print(post['title'])
            print(post['description'])
            print(post['originallink'])
            print(post['link'])
            print(post['pubDate'])

if __name__ == '__main__':
    main()

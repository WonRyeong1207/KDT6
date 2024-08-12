# 인터넷 크롤링: 외부, 내부 링크 모두 저장

# 이건 개인적으로 해보라는 듯? - 쉬는 시간에 보자
from urllib.request import urlopen
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re

# 웹 페이지에서 발견된 내부 링크를 모두 목록으로 만듦
def get_internal_link(bs, incluer_url):
    incluer_url = f'{urlparse(incluer_url).scheme}://{urlparse(incluer_url).netloc}'
    
    internal_links = []
    # "/"로 시작하는 링크를 모두 찾음
    for link in bs.find_all('a', href=re.compile('^(|.*)' + incluer_url + ')')):
        if link.attrs['href']:pass
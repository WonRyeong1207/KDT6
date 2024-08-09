# BeautifulSoup 라이브러리

# 연결 예외처리 하는 방법
# try: 예외가 발생할 수 있는 문장
# except: 예외 발생시 하는 행동
# else: 예외가 발생되지 않았을 때
# finally : 예외 유무와 관련없이 반드시 작동되어야하는 문장
#           예) 파일 닫기

from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.error import URLError

try:
    html = urlopen('http://www.pythonscraping.com/pages/error.html')
except HTTPError as e:
    print(e)
    # 404: not found, 200: found
except URLError as e:
    print('The server could not be found!')
else:
    print('It worked!')

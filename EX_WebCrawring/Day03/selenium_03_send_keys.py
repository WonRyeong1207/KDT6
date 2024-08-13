# selenium으로 로그인 시도 가능, 예전에는 했었다고

# send_keys('text'), send_keys(Keys.ENTER)
# - 키 입력하는 것을 모방할 수 있음.

from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# User Agent 정보 추가
agent_option = webdriver.ChromeOptions()
user_agent_string = 'Mozilla/5.0'
agent_option.add_argument('user-agent'+user_agent_string)

driver = webdriver.Chrome(options=agent_option)
driver.get('https://nid.naver.com/nidlogin.login')

# <input>의 이름이 id를 검색
driver.find_element(By.NAME, 'id').send_keys('id')
driver.find_element(By.NAME, 'pw').send_keys('pw')

# //*[@id="log.login"]
# driver.find_element(By.XPATH, '//*[@id="log.loin"]').click()
driver.find_element(By.ID, 'log.login').click()
time.sleep(3)
driver.quit()

# 네이버에서는 자동입력 방지 문자를 이용해서 로그인이 안되도록 조치를 취함
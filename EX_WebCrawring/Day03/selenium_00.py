from selenium import webdriver
import time
'''
    selenium 4.xx 버전은 chromedriver별도 다운로드 필요 없음
    - selenium 4.23.1
'''

driver = webdriver.Chrome()
driver.get("https://www.selenium.dev/selenium/web/web-form.html")

print(driver.title)
print(driver.page_source)
time.sleep(2)
driver.quit()

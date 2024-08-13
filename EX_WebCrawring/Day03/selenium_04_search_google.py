# selenium API: 구글 검색어 입력 및 검색 결과

from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()
driver.get("https://www.google.com/search?q="	+	'Python')

driver.implicitly_wait(3)

search_results = driver.find_elements(By.CSS_SELECTOR, 'div.yuRUbf')
print()
print(len(search_results))

# Extract ans print the title ans URL of each search result
for result in search_results:
    title_element = result.find_element(By.CSS_SELECTOR, 'h3')
    title = title_element.text.strip()
    print(f"{title}")
    
driver.quit()


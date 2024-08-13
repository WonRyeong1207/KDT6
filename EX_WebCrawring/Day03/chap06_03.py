# 데이터를 파일로 저장

import csv

csvFile = open('./data/test.csv', 'w', encoding='utf-8')

try:
    writer = csv.writer(csvFile)
    writer.writerow(('number', 'number+2', '(numbwe+2)^2'))
    
    for i in range(10):
        writer.writerow((i, i+2, pow(i+2, 2)))
        
except Exception as e:
    print(e)
    
finally:
    # 무조건 실행되어야 하는 구문
    csvFile.close()
    
    
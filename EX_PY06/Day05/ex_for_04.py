# 구구단 2~9단까지 모두 출력

for dan in range(2, 10):
    print("{0:-^37}".format(f"  {dan} 단  "))
    for i in range(1, 10, 3):
        print("{0: <10}".format(f"{dan} * {i} = {dan*i}"), end='    ')
        print("{0: <10}".format(f"{dan} * {i+1} = {dan*(i+1)}"), end='    ')
        print("{0: <10}".format(f"{dan} * {i+2} = {dan*(i+2)}"))
    print("\n")
        

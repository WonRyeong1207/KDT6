# Series/DataFrame에서 사용되는 사용자 정의 함수들


# 함수기능 : DataFrame의 기본정보와 속성 확인 기능
# 함수이름 : checkDataFrame
# 매개변수 : DataFrame 인스턴스 변수병, DataFrame 인스턴스 이름
# 리턴값/반환값 : None

def checkDataFrame(object, name):
    print(f"\n[{name}]")
    object.info()
    print(f"[index] : {object.index}")
    print(f'[columns] : {object.columns}')
    print(f"[ndim] : {object.ndim}")
    
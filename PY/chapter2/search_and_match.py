import re

def search_with_pattern():
    res = re.search('^[A-Z]',' A Python')
    print('[^[A-Z],'' A Python'']의 결과',res) #결과 None (due to space)
    res = re.search('^[A-Z]','A Python')
    print('[^[A-Z],''A Python'']의 결과',res) 
    res = re.search('[^A-Z]','A12 3Python')
    print('[^[A-Z],''A12 3Python'']의 결과',res) 
    res = re.search('[^A-Z]','A Python')
    print('[^[A-Z],''A Python'']의 결과',res) 
    res = re.search('[^A-Z]','APython')
    print('[^[A-Z],''APython'']의 결과',res) 
    res = re.match('[A-Za-z. ]+','kilwon is good. Do you know him? He is GOOD MAN.')
    print(res)
    

def search_basic():
    res = re.search('Python','Python is fun') #Python is fun에서 Python 찾기
    print(res)
    print(res.span()[0],res.span()[1])

def match_basic():
    res = re.match('Python','Python is good')
    print(res)
    print(res.span()[0],res.span()[1])

def have_other_char(): #다른 문자열이 붙어있는 경우
    strs = ['Python','PPython is hard']
    res_search = re.search(strs[0],strs[1]) #search는 찾을 수 있지만
    res_match = re.match(strs[0],strs[1]) #match는 찾지 못함
    print(res_search)
    print(res_match)
if __name__ == '__main__':
    # match_basic()
    # search_basic()
    # have_other_char()
    search_with_pattern()
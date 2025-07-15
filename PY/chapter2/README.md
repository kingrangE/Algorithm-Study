# 2. 정규표현식
- 최초의 컴퓨터 튜링 머신이 탄생할 즈음의 컴퓨터 언어 -> 정규 언어
- 정규 표현식은 정규 언어에서 유래된 것
- 다양한 패턴을 통해 많은 것을 표현하고 패턴을 찾을 수 있다는 장점이 존재
- python에서는 **re**모듈이 정규표현식을 지원한다.

## search, match
### search
- 원하는 패턴을 찾아주는 함수
- 사용법
    - search 함수의 **첫 번째 인수에 문자패턴을 입력**하고 **두 번째 인수에 찾으려는 대상 문자열을 입력**
    - 예시
        ```python
        import re
        res = re.search('Python','Python is fun') #Python is fun에서 Python 찾기
        print(res)
        print(res.span()[0],res.span()[1]) # res.span()[0] : 시작 인덱스, res.span()[1] : 끝 인덱스
        ```
### match
- 원하는 문자열을 찾는 코드
- search와 결과는 같지만, **match의 경우에는 찾고자 하는 문자열 패턴 앞에 다른 문자 혹은 공백 등이 붙어있으면 찾지 못함**
- 사용법
    - search와 같음
        ```python
        import re
        res = re.match('Python','Python is good')
        print(res)
        print(res.span()[0],res.span()[1])
        ```

## 정규표현식 의미
- ^가 \[ 앞에 붙어있다면(^\[\]) **대괄호 안의 표현이 가장 앞에 나온 경우만 찾겠다는 의미**
    - ex, ^[A-Z]라면 알파벳으로 시작하는 것을 찾음
        ```python
        import re
        res = re.search('^[A-Z]',' A Python')
        print(res)#결과 None (due to space)
        res = re.search('^[A-Z]','A Python')
        print(res)#결과 A
        ```
- ^가 대괄호 사이에 있다면(\[^\]) **^의 뒤에 따라오는 표현이 없는 경우만 찾겠다는 의미**
    - ex, \[^A-Z\]라면 대문자가 아닌 문자가 처음 나오는 위치를 찾음
        ```python
        import re
        res = re.search('[^A-Z]','A Python')
        print(res) #[^[A-Z],A Python]의 결과 <re.Match object; span=(1, 2), match=' '>
        ```
- 대괄호 뒤에 +가 붙어있다면(\[ \]+) **\[ \]조건이 어긋날때까지의 문자열을 읽음**
    - ex, \[A-Za-z. \]라면 알파벳 대소문자+점+공백을 찾겠다는 것
        - 이 뒤에 +를 붙이면 해당 조건에 어긋날때까지 찾는다는 것
        ```python
        import re
        res = re.match('[A-Za-z. ]+','kilwon is good. Do you know him? He is GOOD MAN.')
        print(res) #<re.Match object; span=(0, 31), match='kilwon is good. Do you know him'>
        #패턴에 어긋나는 ?가 나오기까지 모든 문자열을 가져옴
        ```
- 패턴과 관련된 특수문자
    1. \ : 문자를 그대로 표현할 때 앞에 붙임
    2. \d : \[0-9\]와 같음
    3. \D : \[^0-9\]와 같음
    4. \w : \[a-zA-Z0-9_\]와 같음
    5. \W : \[^a-zA-Z0-9_\]와 같음
    6. \s : \[\t\n\r\f\v\]와 같음
    7. \S : \[^\t\n\r\f\v\]와 같음
    7. \b : 단어 경계
    8. () : 하나의 그룹을 의미 (ex, (\[\w.\]+)@(\[\w.\]+) 이런식으로)
        - 이때는 각 그룹에 대한 내용과 전체 그룹에 대한 결과를 아래같이 확인할 수 있음
            ```python
            print(res.groups())
            print(res.group(0))
            print(res.group(1))
            ```
## Compile
- 특정 패턴을 반복해야 할 때는 **re.compile()**로 정규표현식의 패턴을 **컴파일된 객체**로 사용하는 방법이 존재함
- 만약, 어떠한 주소창을 인식해야한다면?
    ```python
    import re
    res = re.search('https*:[//\w.]+',".... https://www.google.com")
    print(res.group()) #https://www.google.com
    ```
- Compile 패턴
    - +,*,? 등을 붙여 반복 조건 설정 가능
    1. + : 1개 이상의 문자를 표현, (ex,a+b -> ab,aab,aaaaaaaaab)
    2. * : 0개 이상의 문자를 표현, (ex,a*b -> b,ab,aab,aaaaaaaab)
    3. ? : 0개 또는 1개의 문자를 표현, (ex,a?b -> b,ab)

## findall과 finditer
- 1개 이상의 대상을 검색하고 싶다면 findall,finditer를 사용
### finditer 
- 검색하면 검색 결과로 callable_iterator를 반환함
- 사용 예시
    - 여러개의 링크가 text에 저장되어 있어서 링크 주소만 추출하고 싶은 상황
    ```python
    import re
    text = "Naver : https://www.naver.com, Google : https://www.google.com"
    pattern = re.compile('https*:[//\w.]+')
    m = pattern.finditer(text)
    for item in m:
        st = item.span()[0]
        ed = item.span()[1]
        print(st,ed)
        print(text[st:ed])
    ```
### findall
- findall의 경우에는 위치 정보를 찾는 것이 아닌, 찾은 부분을 리스트로 반환
- 사용 예시
    ```python
    import re
    text = "Naver : https://www.naver.com, Google : https://www.google.com"
    pattern = re.compile('https*:[//\w.]+')
    m = pattern.findall(text)
    for item in m:
        print(item)
    ```
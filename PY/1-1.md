# 문자열
### 공백 제거
- strip() : 양쪽의 공백 제거
- lstrip() : 왼쪽의 공백 제거
- rstrip() : 오른쪽의 공백 제거
- 위의 각 메서드에 "문자"를 넣으면 공백대신 특정 "문자"를 제거함

### 문자 찾기
- find("문자") : 특정 "문자"가 **왼쪽을 기준으로 몇 번쨰에 있는지** 찾기
    - 존재하지 않는 경우 **-1** 반환
- rfind("문자") : 특정 "문자"가 **오른쪽을 기준으로 몇 번쨰에 있는지** 찾기

# Built in type
- List,Tuple,Dictionary,Set 등 파이썬에서 기본으로 제공하는 자료구조

### List
- 여러 원소를 담은 데이터 타입
- \[\] 또는 list()를 사용하여 선언 가능
- 원소 추가 
    - 맨 뒤 : **append('문자')**
    - 특정 위치 : **insert(idx,'문자')**
- 슬라이싱
    - 인덱스 a~b-1까지의 원소를 가져오고 싶다 : **리스트\[a:b-1\]**
    - 첫번째 원소(idx:0) 생략 가능, 마지막 원소 : -1 

### Tuple
- 리스트와 마찬가지로 여러 값을 하나로 묶는 데이터 타입 (원소가 상수여야 하고, 수정 불가)
- \(\)를 사용하거나 마지막 값 뒤에 ,표시
    - ```python
        a = (1,2,3)
        a = 1,2,3,
        ```
### Dictionary
- Map 타입의 자료구조 (Key-Value)
- \{\} 또는 dict()로 선언 가능
- 값 추가
    - ```a[새로운 키] = value```
- 값 수정
    - ```a[기존 키]=value```
- 값 제거
    - ```del a[기존 키]```
    - ```a.pop[기존 키]```
- 키 확인
    - ```'확인하고싶은 키' in a ```
- 값 불러오기
    1. ```a[가져올 키]```
        - 없으면 KeyError
    2. ```a.get(가져올 키)```
        - 없으면 None반환
        - 없을 시, 불러올 값 지정 가능
- zip함수를 사용하면 2개의 리스트를 각 키, 값으로 할당하여 딕셔너리로 선언 가능
    - ```c = dict(zip(['key1','key2'],[1,2]))```

### 집합
- 중복을 허용하지 않는 원소 모음
    - 집합 선언 시, 중복 값을 모두 제거하고 저장함
- {},set()으로 선언 가능
- in, not in 으로 포함여부 확인 가능
- 집합 단위 연산
    - 합집합(union) : |
        - ``` a|b ```
        - ``` set.union(a,b)```
    - 교집합(intersection) : &
        - ```a&b```
        - ```set.intersection(a,b)```
    - 차집합(subtract) : -
        - ```a-b```
        - ```set.difference(a,b)```
    - 배타적논리합(exclusive) :^
        - ```a^b```
        - ```set.symmetric_difference(a,b)```

# 함수와 람다표현식
### 함수
- def로 함수 선언
### 람다
- 파이썬의 익명함수
```python
    func = lambda p : p + 10
    func(1) # 결과 11
```
```python
(lambda p : p + 10)(1) # 바로 사용하기
```
- Lambda함수 응용 (map, filter와 함께 사용)
    - map
        ```python
        list(map(lambda p: p+10,[1,2,3])) #결과 [2,3,4]
        ```
    - filter
        ```python
        list(filter(lambda p: p%2,[2,3,4,5])) #결과 [3,5]
        ```
# 고급 제어
### 반복자
- Class를 반복 가능한 객체로 만들어 Iterator(반복자)로 사용할 수 있음
- 방법
    - Class정의 시, 아래의 두가지 메서드 구현
        1. __iter__
        2. __next__
- 예시
    ```python
    class TestIter:
        def __init__(self,n):
            self.n = n
            self.i = 0
        def __iter__(self):
            return self
        def __next__(self):
            if self.i >= self.n :
                raise StopIteration
            res = self.i
            self.i += 1
            return res
    
    for i in TestIter(3):
        print(i)
    ```
- Iterator로 만든 객체를 iter객체로 만들어 next()로 이용할 수도 있음
    - ```python
        itr = iter(TestIter(3))
        print(next(itr,-1))
        print(next(itr,-1))
        print(next(itr,-1))
        print(next(itr,-1)) #더이상 반복할 곳이 없으면 -1을 반환하도록 하는 것
        ```

### 제너레이터
- yield 키워드를 사용하여 호출자에 값을 전달하는 객체

### 데코레이터
- 디자인 패턴에서 기능을 덧붙이는 것
- 함수에 기능 추가 가능 
- 예시
    ```python
    def print_time(func):
        def wrapper(n):
            stime = time.time()
            res = func(n)
            print('elapse time: {} sec'.format(time.time()-stime))
            return res
        return wrapper

    ## 아래처럼 데코레이터를 통해 함수를 감싸게 되면, 위에서 정의한 함수 실행시간 출력 기능을 더해줄 수 있음
    @print_time
    def cala(n):
        tot = 0
        for i in range(n):
            tot += i
        return tot
    
    ```
### 코루틴
- 2개 이상의 제너레이터가 서로 값을 주고 받으며 실행되는 것
    - 2개 이상의 루틴이 실행됨
- 코루틴은 제너레이터와 다르게 **send** 메서드가 추가됨
    - send를 통해 다른 제너레이터에게 값을 전달해줄 수 있음
- 예시
    ```python
    def sum_coroutine():
        try :
            tot = 0
            while True :
                val = (yield)
                print(val)
                if val == None:
                    return tot
                tot += val
        except Exception as e:
            print(e)
            yield tot

    def total_coroutine():
        while True:
            tot = yield from sum_coroutine()
            print(tot)

    co = total_coroutine()
    co.send(None) # 코루틴 시작 부분 next(co)해도 된다.

    for i in range(1,6):
        co.send(i)
    co.send(None) # 코루틴 종료 부분
    ```

# 클래스
- 하나의 역할을 수행하는데 필요한 메서드와 속성을 은닉시켜 **추상화한 데이터 타입**
- 클래스의 속성을 정의하는 2가지 방법
    1. init 메서드에서 self 키워드를 사용해 저장
    2. Class의 몸체에 Class 속성으로 저장
        - 이 방식을 Class의 Instance끼리 속성을 공유하게 되므로 주의
            - 이때는 문자열,정수같은 단일값에 해당 X, 공유되는 것은 주소를 이용하는 리스트, 딕셔너리 같은 것만 해당
        -  테스트 코드
            ```python
            class calc :
                _x1 = 10
                _x2 = 11
                integer_outer_list= []
                def __init__(self):
                    self.x1 = 1
                    self.x2 = 2
                    self.integer_inner_list = []
                def inner_increasement(self):
                    self.x1+= 1
                    self.x2+= 1
                def outer_increasement(self):
                    self._x1+= 10
                    self._x2+= 10
                    self.integer_outer_list+=[self._x1,self._x2,self.x1,self.x2]
                    self.integer_inner_list+=[self._x1,self._x2,self.x1,self.x2]
                def print_now(self):
                    print("__inner__")
                    print(self.x1, self.x2)
                    print("__outer__")
                    print(self._x1,self._x2)
                    print("__inner_list__")
                    print(self.integer_inner_list)
                    print("__outer_list__")
                    print(self.integer_outer_list)
                
            cal1 = calc()
            cal2 = calc()
            cal1.inner_increasement()
            cal1.outer_increasement()
            cal1.print_now()
            cal2.print_now()
            ## cal2 는 아무런 변화를 주지 않았는데 클래스 변수로 등록한 리스트에서 공유가 일어남
            ```
- Private Method
    - private -> 클래스 메서드에서만 호출 가능한 메서드
    - 생성법 : add prefix(double underscore(__))
        - ex, __add
    - private으로 선언한 메서드에 접근하는 경우 Error Raise
- Static Method
    - static -> 클래스 인스턴스를 생성하지 않고도 사용할 수 있는 메서드 (멤버 속성 등에 접근 불가)
    - 생성법 : ```@staticmethod```
    - static method의 경우 self 인자를 작성해주지 않는다.
- Class Method
    - class method -> 인스턴스 사이에서 공유되는 클래스 속성에 접근 가능한 메서드
    - 정적 메서드처럼 인스턴스를 생성하지 않고도 사용할 수 있는 메서드
- 상속
    - 부모 클래스의 메서드와 속성을 기반으로 새로운 클래스를 만드는 것
    - 보통은 부모 클래스의서 정의한 메서드를 하위 클래스에서 다르게 구현한 뒤, 부모 클래스의 레퍼런스로 제어하는 방식으로 사용
        - 하지만 보통 단순 기능 확장보단 **Composition 기법**을 사용
            - 객체를 클래스의 멤버로 두는 방식
        
    - 부모 클래스의 init에서 속성을 정의하는데, 이를 자식 클래스에서 사용하고 싶다면 자식 클래스에서 ```super().\_\_init\_\_()```으로 명시적으로 호출해야함.

- Abstract Method
    - abstract -> 하위 클래스에서 구현하여 사용해야 하는 메서드
    - 이때, 하위클래스에서 추상 메서드를 구현하지 않으면, 에러가 발생한다.

# 멀티 프로세싱
- 파이썬에서 멀티 스레딩은 하나의 코어만 사용함.
    - 따라서 병렬(parallel)처리가 아닌 병행(concurrent)처리만 가능함.
    - 여러 코어를 사용하지 않는데 이것은 GIL (Global Interpreter Lock) 때문
        - 파이썬에서는 여러 Thread에서 병렬로 접근하면 race condition 문제가 발생해 의도하지 않은 값을 가질 수 있기 때문
            - 이를 막기 위해 하나의 스레드에서만 값을 갱신할 수 있도록 하는 Mutex인 GIL을 동작시키는 것
- Thread를 사용할 수는 있으나, 하나의 코어로 여러 스레드가 동작하는 것에는 한계가 존재함.
    - threading module을 활용하면, 간단하게 멀티스레딩을 구현할 수 있음
        - 하지만 이는 코어의 성능을 최대로 활용하는 multi processing이 아님
- MultiProcessing을 구현하기 위해서는 multiprocess 모듈을 사용
    - 구현법은 Pool 클래스, Process 클래스 사용 2가지 방식이 존재
### Pool을 기반으로 Mutli Processing 구현
- Pool을 사용하면 구동하고 싶은 Process수만큼 제약없이 한 번에 구동이 가능함.
- 얘제 : Pool로 Consumer와 Producer를 구현
    - 이때, Producer가 무조건 먼저 작동한다는 보장이 없기 때문에 Consumer가 먼저 작동하는 경우에 대해 대처할 수 있도록 고려하여 코드를 작성해야 함.
    ```python
    from multiprocessing import Pool, Manager
    class HandleParam:
        def __init__(self,name,queue,lock,cond,role):
            self.name = name # 프로세스 이름
            self.queue = queue # 다른 프로세스와 통신에 사용할 큐
            self.lock = lock # 다른 프로세스와 통신에 사용할 락
            self.cond = cond # 다른 프로세스와 통신에 사용할 condition 변수
            self.role = role # 프로세스의 역할

    # 프로세스가 구동하면 실행할 handler
    def handler(param : HandleParam):
        print("process name {}".format(param.name))
        queue = param.queue
        cond = param.cond

        if 'producer' == param.role :
            queue.put('hello')

            cond.acquire()
            cond.notify()
            cond.release()
        elif 'consumer' == param.role :
            cond.acquire()
            cond.wait_for(lambda: not queue.empty()) # 아직 queue에 메시지가 없다면 대기
            cond.release()
            
            print('consuming messages {}'.format(queue.get()))

    num_process = 2
    pool = Pool(processes = num_process)

    # 두 프로세스가 공유할 lock, cond, queue를 Manager 객체를 이용하여 가져옴
    manager = Manager()
    lock = manager.lock()
    cond = manager.Condition()
    queue = manager.Queue()

    process_list = []
    # HandlerParam 객체는 여러 프로세스에 하나씩 분배가능하도록 배열에 담겨있어야 함.
    process_list.append(HandleParam('producer',queue,lock,cond,'producer'))
    process_list.append(HandleParam('consumer',queue,lock,cond,'consumer'))

    pool.map(handler,process_list) # 각 프로세스를 handler에 담음

    pool.close() # pool 객체 종료
    pool.join() # pool 객체 종료까지 대기
    ```
### Process를 기반으로 구현
- Process는 개별 프로세스를 별도로 제어해야 할 때, 사용
- 예제  : Process를 사용하여 개별적으로 제어
    ```python 
    from multiprocessing import Process
    import multiprocessing as mp
    import collections

    def handler(name,queue,cond):
        if 'producer' == name :
            queue.put('hello')
        else : 
            cond.acquire()
            cond.wait_for(lambda: not queue.empty())
            cond.release()

            print('consuming message',queue.get())

    roles = ['producer','consumer']
    processes = collections.defaultdict(None)

    lock = mp.lock()
    cond = mp.Condition(lock)
    queue = mp.Queue()

    for role in roles :
        # Pool이랑 다르게 각 프로세스를 직접 하나씩 등록, 각자 시작
        processes[role] = Process(target=handler,args=(role,queue,cond))
        processes[role].start()

    for role in roles:
        # join 역시 개별적으로 진행
        processes[role].join()
    ```
    - 대부분의 코드가 Pool 사용과 유사하나, 큰 차이는 개별적으로 Start,Join을 제어해주어야 한다는 것
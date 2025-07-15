# 핵심 자료구조
## 3.1 스택
- 가장 흔히 쓰이는 자료구조
- LIFO (후입 선출 구조)
- 동작 방식
    - Stack 자료구조는 push,pop,top,size,empty 등의 기능을 제공
        1. push : 스택에 값을 저장
        2. pop : 가장 최근에 저장한 값을 스택에서 제거
        3. top : 가장 최근에 저장한 값을 반환
        4. size : 스택에 저장된 값의 개수를 반환
        5. empty : 스택이 비어있는지 확인
- 구현 코드
    ```python
    class stack :
        def __init__(self):
            self.data = []
        def push(self,x):
            self.data.append(x)
        def pop(self):
            return self.data.pop()
        def size(self):
            return self.data.__len__()
        def empty(self):
            return 1 if self.data.__len__() == 0 else 0
        def top(self):
            return self.data[-1]
    ```

### 3.2 큐
- 들어온 순서대로 처리하는 것을 보장하는 자료구조
- FIFO (선입 선출 구조)
- 동작 방식
    - enqueue,size,dequeue 등의 기능을 제공
        1. enqueue : queue의 앞부분에 원소를 추가하는 연산
        2. dequeue : queue의 뒷부분의 원소를 제거하는 연산
        3. size : queue에 들어있는 원소를 확인
    - 기타(필수적이지 않음) : peek, empty 등
        1. peek : queue의 가장 앞에 위치한 원소 **확인**
        2. empty : queue가 비어있는지 확인
- 구현코드
    - 원형 큐 : 선형 방식으로 큐를 운영
        - front가 가리키는 위치에 enqueue
        - rear가 기리키는 위치에서 dequeue
        - front-rear가 같은 위치를 가리킨다면 큐가 빈상태
        - 구현
            ```python
            class CircleQueue:
                def __init__(self,k: int):
                    self.n = k+1 
                    delf.data = [0]*(self.n)
                    self.front = 0
                    self.rear = 0
                def dequeue(self) -> bool:
                    if self.isEmpty():
                        return False
                    self.rear = (self.rear+1)%self.n
                    return True
                def isFull(self):
                    if (self.front+1)%self.n == self.rear:
                        return True
                    return False
                def isEmpty(self):
                    if self.front == self.rear:
                        return True
                    return False
                def enqueue(self,x: int) -> bool:
                    if self.isFull():
                        return False
                    self.data[self.front] = x
                    self.front = (self.front+1)%self.n
                    return True 
            ```

### 3.4 연결리스트
- 원형큐의 문제 : 고정 크기
    - 실무에서는 고정 크기를 가정할 수 있는 경우가 많지 않음
    - 이떄 사용할 수 있는 것 : 연결 리스트(linked list)
- 종류
    - 단일 연결 리스트 : 값, 포인트를 가지는 연결리스트
        - 구현
            ```python
            class LinkedList :
                def __init__(self,size):
                    self.size = size
                    self.num = 0
                    self.head = Node(None)
                def insert(self,value):
                    if self.num >= self.size:
                        return False
                    node = Node(value)
                    node.next = self.head.next
                    self.head.next = node
                    self.num += 1
                    return True
                def traverse(self):
                    cur = self.head.next
                    while cur :
                        print(cur.data,end=' ')
                        cur = cur.next
                    print()
                def remove(self,value):
                    pre = self.head
                    cur = self.head.next
                    while True:
                        if cur.data == value:
                            pre.next = cur.next
                            self.num -= 1
                            return True
                        pre = cur
                        cur = cur.next
                    return False
            ```
    - 이중 연결 리스트
        - 이중 연결 리스트는 연결링크가 앞, 뒤로 2개라는 것이 가장 큰 차이점
        - 또한 단일 연결리스트가 무조건 head에서 시작했던 것과 달리, 이중 연결 리스트는 head와 tail을 가지고 있어 앞,뒤 어디서든 순회를 할 수 있다.
        - 구현
            ```python
            class DoubleLinkedList:
                def __init__(self,size):
                    self.size = size
                    self.num = 0
                    self.head = Node(None)
                    self.tail = Node(None)
                    self.head.next = self.tail
                    self.tail.pre= self.head
                def insert(self,value):
                    if self.num >= self.size:
                        return False
                    node = Node(value)
                    node.next = self.head.next
                    self.head.next.pre = node
                    self.head.next = node.next
                    node.pre = self.head
                    self.num += 1
                    return True
                def traverse(self):
                    cur = self.head.next
                    while cur != self.tail:
                        print(cur.data,end=' ')
                        cur = cur.next
                    print()
                def remove(self,value):
                    cur = self.head.next
                    while cur:
                        if cur.data == value :
                            cur.next.pre = cur.pre
                            cur.pre.next = cur.next 
                            self.num -= 1
                            return True
                        cur = cur.next
                    return False          
            ```            
### 복사와 레퍼런스의 차이
- 파이썬에서 할당을 연산할 때, 기본 변수 타입은 값이 복사된다.
- 하지만, 변수가 객체인 경우에는 값을 복사하는 것이 아닌 레퍼런스(Pointer)를 갖게 된다.
- EX)
    1. 기본 변수 타입인 경우
        ```python
        val1 = 1
        val2 = val1
        print(val1 is val2) #True
        print(val1 == val2) #True
        val1 = 2
        print(val1) #2
        print(val2) #1
        print(val1 is val2) #False
        print(val1 == val2) #False
        ```
    2. 레퍼런스 타입인 경우
        ```python
        class Node :
            def __init__(self,data):
                self.data = data
                self.next = None
        val1 = Node(1)
        val2 = val1
        val1.data = 2
        print(val1.data) #2
        print(val2.data) #2
        print(val1 is val2) #True
        print(val1 == val2) #True
        ```

### 3.5 해시, 맵
- key-value mapping하는 자료구조 
    - mapping되어 있기 때문에 원소의 개수와 상관없이 값을 찾는데 항상 $O(1)$의 시간복잡도를 갖는다.
- 이런 형태의 자료구조는 대부분 hash를 사용하여 데이터를 빠르고 효율적으로 관리함.
    - 해시 종류
        1. hash set
            - 집합의 성격을 가지고 있어 중복을 허용하지 않음.
            - 해시 맵으로 구현 가능
        2. hash map(hash table)
            - 해시맵은 add,delete,search를 거의 $O(1)$의 속도로 수행할 수 있는 자료구조
            - 배열과 리스트 특징을 모두 갖고 있어 구현이 쉬움
### Hash Map
- HashMap의 경우 HashFunction과 HashTable로 이루어져있음
    - 이때 HashFunction은 Key를 넣으면 Table의 어디에 넣으면 되는지 Index를 제공해주는 역할을 한다.
        - 만약 hash function의 결과가 같다면? Crash!
            - 해결 방법 2가지
                1. 개방 주소법 (open address)
                2. 체이닝 (chaining)
### 1. 개방 주소법
- Crash가 발생하면, 충돌한 bucket부터 선형적으로 증가하며 비어있는 버킷을 찾아 넣음
- 순차적으로 검색하므로 아래와 같이 부르기도 함
    1. linear probing
    2. 선형 탐사
- 단점 
    1. 빈 곳을 빨리 찾지 못하면 많은 시간 소모 가능
    2. 자료의 개수를 예측하고 미리 버킷을 준비해야 하므로 예측 가능한 입력에만 효과적
    - 즉 자료의 특성을 알고 있을 때 효율적 (사진 저장, disk 매체에 적합)
- 구현 코드
    ```python
    class HashMap:
        def __init__(self,size,hash_func):
            self.table_size = size
            self.table = [Bucket() for i in range(size)]
            self.hash_func = hash_func
            self.n = 0
        def __next(self,key)->int:
            return 1
        def insert(self,key,value):
            idx = start = self.hash_func(key)%self.table_size
            while self.table[idx].state == 'FILLED': #이미 차있으면
                idx = (idx + self.__next(key)) % self.table_size# 다음 인덱스
                if idx == start : #테이블 이미 가득차서 돌아옴
                    return False
            self.table[idx].state = 'FILLED'
            self.table[idx].key = key
            self.table[idx].value = value
            self.n += 1
            return True
        def find(self,key):
            idx = start = self.hash_func(key)%self.table_size
            while self.table[idx].state != 'EMPTY':
                if self.table[idx].state == 'FILLED' and self.table[idx].key == key:
                    return index
                idx = (idx + self.__next(key)) % self.table_size
                if index == start :
                    return -1
            return -1
        def get(self,key) ->int:
            index = self.find(key)
            if index == -1:
                return None
            return self.table[index].value
        def remove(self,key)->bool:
            idx = self.find(key)
            if -1 == idx:
                return False
            self.table[idx].state = 'DELETED'
            self.n -= 1
            return True
        def remove_at(self,pos)->bool:
            if pos<0 or pos>=self.table_size or self.n ==0 :
                return False
            if self.table[pos].state != 'FILLED':
                return False
            # FILLED면
            self.table[pos].state = 'DELETED'
            self.n -=1
            return True
        
    ```
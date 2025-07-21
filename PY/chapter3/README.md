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

## 3.2 큐
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

## 3.4 연결리스트
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

## 3.5 해시, 맵
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
### 2. 체이닝
- 구현 코드
    ```python
    class HashMap:
        def __init__(self,size,hash_func):
            self.table_size = size
            self.table = [[] for i in range(size)]
            self.hash_func = hash_func
            self.n = 0
        def insert(self,key,value):
            idx = self.hash_func(key)%self.table_size
            self.table[idx].insert(0,Bucket(key,value))
            return True
        def find(self,key):
            idx = self.hash_func(key)%self.table_size
            for node in self.table[index]:
                if node.key == key:
                    return True
            return False
        def remove_at(self,key):
            index = self.hash_func(key)%self.table_size
            i = 0
            found = False
            for node in self.table[index]:
                if node.key == key:
                    found = True
                    break
                i+=1
            if found :
                del self.table[index][i]
                return True
            return False
    ```

- table의 원소 -> Bucket이 아닌 빈 리스트  
    - 이중 연결 리스트를 의미

- insert
    - 개방주소법보다 간단한데, 이것은 충돌 발생 여부와 관계없이 값을 Bucket에 넣고, Bucket을 리스트의 처음 위치에 추가하면 되기 때문
- find
    - hash함수를 통해 나온 index번째 위치에 있는 리스트를 순회하며 key를 찾음 
- remove_at
    - find와 동일하게 값을 찾고, 찾았으면 리스트의 해당 인덱스에 있는 Bucket을 제거한다.

### 성능 분석
| 연산 | 평균 시간 복잡도 | 최악 시간 복잡도 |
| 삽입 | O(1), 상수 시간 | O(N), 선형 증가 시간 |
| 검색 | O(1), 상수 시간 | O(N), 선형 증가 시간 |
| 삭제 | O(1), 상수 시간 | O(N), 선형 증가 시간 |

## 3.6 트리
- 한 곳에서 여러 형태로 파생되는 계층적 구조를 시각화하는데 흔히 쓰이는 알고리즘
    - 트리로 흔히 표현하는 것 : 가계도

- 대표적인 트리 구조
    - 이진 트리
        - 부모 노드에서 파생된 왼쪽 자식 노드와 오른쪽 자식 노드가 존재하는 트리
            - 한 노드가 최대 2개의 자식 노드를 가질 수 있다는 뜻
            - 부모 노드처럼 트리에서 가장 먼저 출현하는 노드 : **루트 노드**
    - 이 외에도 한 부모가 파생할 수 있는 자식 수에 따라 **삼항 트리, 다항 트리** 등이 있음
        - 삼항 트리 : 자식 3개, 다항 트리 : 자식 수 제한 X
- 본 책에서는 이 중 가장 많이 사용되는 이진 탐색 트리(binary search tree,BST)를 통해 설명할 예쩡

### 용어
- 부모 노드(루트 노드) : 파생되기 시작하는 1번 노드
- 자식 노드 : 파생된 노드
- 잎 노드, 단말 노드 : 자식 노드가 없는 노드
- 레벨 : 루트 노드로부터의 거리
- 트리의 깊이 : 트리의 최대 레벨
- 내부 노드 : 루트 노드나 단말 노드가 아닌 노드
- 형제 노드 : 부모가 같은 노드
- 완전 이진 트리 : 모든 내부 노드가 2개의 자식을 가진 트리
- 포화 이진 트리 : 각 레벨에 빠지는 노드가 없는 트리
- 이진 탐색 트리 : 탐색 목적으로 사용하는 이진 트리

- 이진 트리와 이진 탐색 트리의 차이점
    - 이진 탐색 트리의 규칙
        1. 각 노드가 가지고 있는 양의 정수나 문자, 문자열을 키로 사용할 수 있으며 키는 값 비교에 사용함.
        2. 왼쪽 자식 노드의 키값 < 부모 노드의 키 값 < 오른쪽 자식 노드의 키 값
    - 이 규칙을 통해 트리를 탐색할 때, 왼쪽으로 갈 지, 오른쪽으로 갈 지 결정할 수 있음

### 동작 방식

- insert
    - 검색에 사용할 값을 트리에 추가
        - 예시 : 7,5,10,6을 순서대로 추가한다는 가정
            1. 아무 원소가 없는 트리에 7이라는 값을 가진 노드 추가
                - 처음 생성하는 노드이므로 이 노드가 루트 노드
            2. 5를 값으로 가진 노드를 트리에 추가
                - 규칙에 따라 5는 7보다 작으므로 7의 왼쪽 자식 노드로 추가된다.
            3. 10을 값으로 가진 노드를 트리에 추가
                - 규칙에 따라 10은 7보다 크므로 7의 오른쪽 자식 노드로 추가된다.
            4. 6을 값으로 가진 노드를 트리에 추가
                - 6은 7보다 작으므로 루트노드의 왼쪽에 추가
                - 그러나, 왼쪽엔 5가 존재, 6이 5보다 크므로 5의 오른쪽 자식 노드로 추가
- find
    - 트리의 노드가 가진 키를 비교하여 특정 키를 지닌 노드를 찾음
    - 이진 탐색 트리에서 값을 찾을 땐, 현재 방문한 노드보다 크면 오른쪽으로 이동, 작으면 왼쪽으로 이동
- traverse
    - 모든 노드를 순회하는 함수
        - 전위, 중위, 후위, 단계
        1. 전위 순회
            - 트리의 여러 값을 순회할 때, 현재 방문한 노드를 가장 먼저 처리하는 방식
            - 따라서 현재 노드 -> 왼쪽 자식 노드 -> 오른쪽 자식 노드 순서로 모든 노드를 방문
        2. 중위 순회
            - 왼쪽 자식 노드를 먼저 방문하여 노드를 처리하고 다음으로 오른쪽 자식 노드를 방문하는 방식
            - 따라서 왼쪽 자식 노드 -> 현재 노드 -> 오른쪽 자식 노드 순서
        3. 후위 순회
            - 현재 방문하는 노드를 가장 마지막에 처리하는 순회 방식
            - 왼쪽 자식 노드 -> 오른쪽 자식 노드 -> 현재 노드 순서
        4. 단계 순회
            - 트리의 레벨 단위로 순회 
            - Top-Down Approach
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
- remove
    - 노드를 삭제하는 함수
        - 삭제의 경우의 수
            1. 말단 노드 삭제
            2. 자식이 1개인 노드 삭제
            3. 자식이 2개인 노드 삭제
    - Detail
        1. 말단 노드 삭제
            - 상위 노드와의 링크를 끊어 쉽게 삭제 가능
        2. 자식이 1개인 노드 삭제
            - 유효한 자식 노드를 부모 노드에 연결한 다음 삭제
        3. 자식이 2개인 노드
            - 왼쪽, 오른쪽 어느 방향으로 이동할지 정해야함
                - ex) 왼쪽 노드로 이동해야하는 경우
                    1. 삭제할 노드의 왼쪽 자식 노드로 이동
                    2. 오른쪽에 자식노드가 없을 때까지 오른쪽으로 이동
                    3. 발견 시, 오른쪽 자식 노드로 삭제할 노드의 오른쪽 자식 노드 지정
                    4. 삭제할 노드의 왼쪽 자식 노드를 삭제할 노드 부모의 왼쪽 자식 노드로 연결
### 구현 예시
``` python
class Tree :
    # 루트 노드는 None으로 초기화
    def __init__(self):
        self.root = None
    
    # 노드 삽입
    def insert(self, node):
        if not self.root:
            self.root = node
            return 
        
        def dfs(cur):
            if not cur:
                return node
            if cur.value < node.value : # 이진 트리 조건 (부모 < 오른쪽 자식 노드)
                cur.right = dfs(cur.right) # 오른쪽 자식이 없을 때까지 계속 이동 
            else : 
                cur.left = dfs(cur.left) # 이진 트리 조건을 만족하지 못하는 경우 left로 계속 이동
            return cur

        dfs(self.root)

    def find(self,key):
        cur = self.root
        while cur:
            if cur.value == key:
                break
            if cur.value < key:
                cur = cur.right
            else :
                cur = cur.left
        return cur if cur.value == key else None
    
    def remove_node(self,key):
        def dfs(node):
            if not node :
                return None
            if node.value == key:
                if not node.left and not node.right:
                    return None
                if not node.left or not node.right:
                    return node.left or node.right
                cur = node.left
                while cur.right:
                    cur = cur.right

                cur.right = node.right
                return node.left
            elif node.value < key :
                node.right = dfs(node.right)
            elif node.value > key :
                node.left = dfs(node.left)
            return node
            
        return dfs(self.root)

    def pre_order(self,cur):
        print(cur.value, end=' ') # 본 노드
        if cur.left : 
            self.pre_order(cur.left) # 왼쪽 자식
        if cur.right :
            self.pre_order(cur.right) # 오른쪽 자식
    def in_order(self,cur):
        if cur.left : 
            self.pre_order(cur.left) # 왼쪽 자식
        print(cur.value, end=' ') # 본 노드
        if cur.right :
            self.pre_order(cur.right) # 오른쪽 자식
    def post_order(self,cur):
        if cur.left : 
            self.pre_order(cur.left) # 왼쪽 자식
        if cur.right :
            self.pre_order(cur.right) # 오른쪽 자식
        print(cur.value, end=' ') # 본 노드
    def level_order(self,cur):
        q = collections.deque([cur])
        while q :
            cur = q.popleft()
            print(cur.value,end=" ")
            if cur.left:
                q += cur.left
            if cur.right :
                q += cur.right
```
### 성능 분석
- Binary Search Tree 성능
| 연산 | 평균 시간 복잡도 | 최악 경우의 시간 복잡도 |
| 삽입 | $O(logN)$ | $O(N)$ |
| 검색 | $O(logN)$ | $O(N)$ |
| 삭제 | $O(logN)$ | $O(N)$ |

- Balanced Binary Search Tree의 성능
| 연산 | 평균 시간 복잡도 | 최악 경우의 시간 복잡도 |
| 삽입 | $O(logN)$ | $O(logN)$ |
| 검색 | $O(logN)$ | $O(logN)$ |
| 삭제 | $O(logN)$ | $O(logN)$ |

### 트라이(trie)
- Tree의 한 종류로 prefix tree라고도 부름
    - 이진 트리와 차이점
        1. 자식 노드의 개수에 제한이 없음
        2. 키 값의 대소 관계를 파악하여 값을 찾는 것이 목적이 아님
            - 노드를 추가할 때, 키 값의 크기에 따라 노드의 위치를 고려할 필요가 없음
- 사용하는 경우
    - 문자, 문자열을 저장하고 빠르게 검색하는 용도
        - EX) 포털사이트 검색어 자동완성
- 동작 방식
    - insert
        - trie에 값을 추가하는 함수
            - EX) bst, best, bestow, black 을 추가한다 가정
                1. bst를 b,s,t 각각을 노드로 연결
                2. best는 b가 겹치므로 b 노드 아래에 새로운 자식 노드 e,s,t를 각각 순서대로 연결 
                3. bestow는 best가 겹치므로 b -> e -> s -> t 까지 같고, 여기에 이어 o,w 노드를 연결
                4. black도 동일함
                - Q) best와 bestow는 best부분이 동일한데, be라고 입력했을 때, best가 어떻게 나오냐?
                    - 문자 검색을 멈추도록 하기 위해 EOW(End of Word)라는 표시를 노드에 남김
                    - EOW를 이용하여 이전에 입력했던 완전한 문자열인지를 판단할 수 있음
    - search
        - 원하는 문자를 검색하는 함수
            - EX) 검색창에 b만 입력해도 연결된 키워드가 출력되도록 하려면
                1. b검색 시, b값이 저장된 노드 방문
                2. b에 붙은 자식 노드들을 Queue와 같은 자료구조에 저장
                3. 3개의 노드를 각각 방문하여 반복 
                4. EOW를 만나게되면 지금까지 방문한 노드의 문자를 모아 결과 리스트에 추가
- 구현 코드
    ``` python
    class TrieNode:
        def __init__(self,letter):
            self.letter = letter
            self.child = {}
            self.eow = False
    class Trie:
        def __init__(self):
            self.root = TrieNode(None)
        def insert(self,word: str) -> None:
            cur = self.root
            for ch in word :
                if ch not in cur.child:
                    cur.child[ch] = TrieNode(ch)
                cur = cur.child[ch]
            cur.eow = True
        def search(self, word: str) -> bool:
            cur = self.root
            for ch in word :
                if ch not in cur.child:
                    return False
                cur = cur.child[ch]
            return True if True == cur.eow else False
        def starts_with(self,prefix: str) -> bool:
            cur = self.root
            for ch in prefix:
                if ch not in cur.child:
                    return False
                cur = cur.child[ch]
            return True
        
        def get_starts_with(self,prefix: str) -> List:
            cur = self.root
            for ch in prefix:
                if ch not in cur.child:
                    return None
                cur = cur.child[ch]
            if not cur :
                return None
            q = collections.deque([(cur,prefix)])
            res = []

            while q:
                cur, word = q.popleft()
                if cur.eow:
                    res += word
                for node in cur.child.values():
                    q.append((node, word+node.letter))
            return res
    ```

### 기수 트리(radix tree)
- 기본적으로 이진 탐색 트리와 동일
    - 차이점 
        - 이진 탐색 트리 : 노드의 키값을 비교해 노드를 추가
        - 기수 트리 : 키 값을 비교하지 않고, 키가 가진 **비트** 값을 비교해 비트가 0이면 왼쪽, 1이면 오른쪽에 노드 추가
- 비트 값을 비교할 땐 루트 노드를 제외하고 다음 레벨부터 시작
    - Level 2에서는 LSB(최하위 비트: 가장 작은 숫자)를 비교
    - Level 3에서는 LSB에서 한 bit 이동한 bit(오른쪽에서 두 번째 비트)를 비교
    - Level 이 깊어질 수록 MSB로 1칸씩 이동한 Bit로 비교
        - MSB, LSB -> 정수를 이진수(bit)로 변환했을 때, 가장 앞에 있는 비트가 MSB 가장 뒤에 있는 비트가 LSB
    - EX) 8bits를 사용하는 기수 트리 
        - Max depth : 8+1 (8bit + 1bit(root node))
        - Max Number of Node : $2^8$
- 기수 트리는 같은 값을 중복하여 저장하지 않음
- 사용하는 경우
    - 값이 편향된 경우에도 쏠림 현상 방지가 가능함
        - 기수 트리는 값을 level별로 2진수의 특정 bit로 비교하기 때문
    - bit는 값이 증가할 때, 0->1로 toggle되는 특징이 존재, 이를 잘 활용한 것이 기수 트리
    - linux kernel의 cache 관리 module에 기수 트리를 사용(검색 성능 $O(logN)$ 유지)
- 기수 트리의 단점
    - 노드 삭제 시, 계산 비용이 크다
        - Due to, 절대 크기를 비교하는 것이 아님
        - To solve this problem, 삭제를 하기보다 유효하지 않다는 표시를 한 뒤, 이후 한 번에 재정렬 과정 수행
- 구현 코드
    ``` python
    class RadixNode:
        def __init__(self,value):
            self.value = value
            self.left = None
            self.right = None
    class RadixTree:
        def __init__(self,m):
            self.root = None
            self.m = m
            self.fmt = '{:0'+str(self.m)+'b}'
        def __deinit__(self):
            ...
        def convert_bits(self,key):
            assert(len('{:b}'.format(key))<= self.m)
            return self.fmt.format(key)
        def find(self,key):
            cur = self.root
            bits = self.convert_bits(key)

            while cur and i>= 0 :
                if cur.value == key :
                    break
                if '0' == bits[i]:
                    cur = cur.left
                else :
                    cur = cur.right
                i -= 1
            return cur if cur.value == key else None
        def insert(self,node):
            if not self.root:
                self.root= node
                return
            def dfs(cur,bits,i):
                if not cur:
                    return node
                if '0' == bits[i]:
                    cur.left = dfs(cur.left,bits,i-1)
                else:
                    cur.right = dfs(cur.right, bits, i-1)
                return cur
            bits = self.convert_bits(node.value)
            dfs(self.root,bits,len(bits)-1)
    ```

## 3.7 힙
- Queue는 FIFO(선입선출) 방식으로 데이터를 처리하는 대표적인 자료구조
    - 먼저 들어온 자료가 먼저 처리되는 순서가 보장된 자료구조
    - 단점 
        - 최단 거리 판별, 최장 거리 판별 시 들어온 순서가 아닌 **값의 크고 작음**에 따라 순서를 정해야 할 때 사용 어려움
            - 이를 해결하는 것이 priority queue

- priority queue
    - 최솟값을 기준으로 Queue에서 값을 내보내는 자료 구조
    - 내부에서 연결리스트 혹은 트리를 사용하여 구현할 수 있음
        - BUT, 연결리스트 사용 시, 자료구조 특성 상 성능이 그렇게 좋지 못함
    - 가장 성능이 좋게 구현하려면 **Heap** 자료구조를 사용한다.
- Heap
    - 완전 이진 트리(full binary tree)의 일종, priority queue를 Optimal 성능으로 구현하는 것이 목적
    - 특징
        1. 부모는 두 자식보다 값이 커야함
        2. 루트의 값이 가장 커야함
        3. Full binary tree의 일종
    - **루트 노드가 항상 가장 작은 값임을 보장**하는 자료구조
    - **Heap**으로 우선순위 큐를 구현하면 가장 작은 값의 검색을 $O(1)$의 시간 복잡도로 수행 가능
        - 따라서 Heap으로 구현한 priority queue가 최솟값 획득시 가장 유리
    - 종류
        1. 최대힙 : 가장 큰 값이 트리의 위쪽에 위치
        2. 최소힙 : 가장 작은 값이 트리의 위쪽에 위치
### 동작 방식
1. insert+upheap
    - 값을 추가하는 과정
    - EX, [5,3,7,4,10]을 힙에 순서대로 추가
        1. heap이 빈 상태이므로, 가장 먼저 추가되는 원소 5가 root node
        2. 3을 heap tree의 단말 노드에 추가
        3. upheap을 해야 하지만, 추가한 노드의 값(3)이 부모 노드(5)보다 작으므로 upheap을 수행하지 않음
        4. 7을 추가, 7이 부모 노드(5)보다 크므로 자리를 교환(swap:upheap)
        5. 4를 추가, 4가 부모 노드(3)보다 크므로 자리를 교환(swap:upheap), 4가 루트 노드(7)보다 작으므로 swap 수행 X
        6. 10을 추가, 10이 부모 노드(4)보다 크므로 자리를 교환, 10이 루트 노드(7)보다 크므로 한 번 더 교환
2. extract+downheap
    - 값을 제거하는 과정
    - EX, [10,7,5,4,3]을 순서대로 제거
        1. 최상단 노드(루트노드(10))를 힙에서 제거
        2. 말단 노드(4)를 루트로 이동
            - 이때 downheap 수행 (heap 성질 유지를 위해 값을 swap 하며 이동하는 과정)
            - downheap에서 4와 7을 swap, 4는 3보다 크므로 swap(downheap) 종료
        3. 위 과정을 반복
- 한 노드의 downheap과 upheap 모두 최대 트리 레벨만큼 연산이 수행되므로 downheap과 upheap의 시간 복잡도는 모두 $O(logN)$

### 구현 코드
``` python
class Node:
    def __init__(self,value):
        self.value = value
class HeapQueue:
    def __init__(self):
        self.num = 0
        self.nodes = []
    def __deinit__(self):
        ...
    def insert(self,value):
        self.nodes.append(Node(value))
        self.num+=1
        self.upheap()
    def __swap(self, lnode, rnode):
        lnode.value, rnode.value = rnode.value, lnode.value
    def upheap(self):
        cur_idx = self.num - 1
        cur = self.nodes[cur_idx]

        while cur_idx >0 :
            par_idx = (cur_idx - 1)//2
            par = self.nodes[par_idx]
            if cur.value <= par.value:
                break
            self.__swap(par,cur)
            cur = self.nodes[par_idx]
            cur_idx = par_idx
    def downheap(self,cur_idx):
        cur = self.nodes[cur_idx]
        while cur_idx < self.num // 2 :
            chd_idx = 1 if cur_idx == 0 else ((cur_idx + 1) *2 -1)
            if chd_idx + 1 < self.num and self.nodes[chd_idx].value < self.nodes[chd_idx + 1].value:
                chd_idx += 1
            if cur.value < self.nodes[chd_idx].value :
                self.__swap(cur,self.nodes[chd_idx])
            cur = self.nodes[chd_idx]
            cur_idx = chd_idxcode
    def extract(self):
        if 0 == self.num:
            return None
        node = self.nodes[0]
        self.nodes[0] = self.nodes[self.num -1]
        del self.nodes[self.num -1]
        self.num -= 1

        if self.num > 0:
            self.downheap(0)
        return node.value
```
## 3.8 그래프
- Vertex(정점)과 Edge(간선)으로 구성
    - Vertex
        - 그래프의 구성 요소
        - degree(차수)가 존재 (연결된 정점의 수)
    - Edge
        - Vertex를 연결하는 것(관계를 의미)
- 종류
    1. 유향 그래프(directed graph)
        - 방향성이 존재하는 Graph
        - Vertex간의 이동이 양방향이 아닌 그래프
        - 이때는 방향성이 존재하므로 Degree도 2가지
            1. in-degree (내향 차수)
                - 자기 자신으로 들어오는 degree
            2. out-degree (외향 차수)
                - 밖으로 나가는 degree
        - 도달 가능성 판단 문제(탐색 알고리즘), 강한 결합 요소 찾기, 위상/역위상 정렬, 임계 작업 구하기 등에서 사용

    2. 무향 그래프(undirected graph)
        - Vertex간의 연결에 방향성이 없는 그래프
            - 이동이 양방향으로 가능하다는 의미
        
    3. 가중치 그래프(weighted graph)
        - Vertex사이에서 이동하는데 소모되는 비용을 표시한 그래프
        - 최단거리, 공정 문제 등의 분야에서 문제 해결 가능
    4. 유향 비순환 그래프(directed acylic graph,DAG)
        - 방향 그래프에 순환 참조가 없는 그래프
        - 푸는 대부분의 문제가 이 그래프를 활용하는 문제
- 동작 방식
    - 기능
        1. 정점 추가
        2. 정점 탐색
        3. 정점 간 거리 파악
        4. 순환 참조 여부 파악
    - 본 내용에서는 위의 기능 중 **정점 추가**를 5가지 표현 방식으로 살필 예정
        - 인접 행렬, 인접 리스트, 셋, 인접 딕셔너리, 가중치 그래프

### 인접 행렬
- 무향 그래프는 하나의 행과 열이 모두 동일한 값을 가짐
    - 자기 자신과 연결된 것은 없으므로 대각 원소는 모두 0
- 장점
    - **단 한 번의 첨자 연산(index 연산)으로 두 정점 간 연결 여부를 확인**할 수 있다는 것
- 단점
    - 연결되지 않은 관계까지 0으로 표현하므로 **공간을 많이 사용**
- 코드 구현
    ``` python
    n = 5
    edges = [[1,2],[2,3],[2,4],[2,5]]

    adj_mat = [[0]*n for _ in range(n)]

    for u,v in edges:
        adj_mat[u-1][v-1] = 1
        adj_mat[v-1][u-1] = 1
    
    for line in adj_mat:
        print(line)
    ```

### 인접 리스트
- 인접 행렬과 정반대의 장단점
    - 장점 : 불필요한 공간 사용 X
    - 단점 : 연결을 찾을 때, 선형 탐색 해야함

- 구현
    ```python
    adj_list = [[] for _ in range(n+1)]

    for u,v in edges : 
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    for row in adj_list:
        print(row)
    
    def is_edge_exist(u,v):
        for vertex in adj_list[u]:
            if v == vertex :
                return True
        return False
    ```

### 셋
- 인접 행렬, 인접 리스트보다 효과적인 자료구조
- 구현
    ``` python
    graph = [set() for _ in range(n+1)]

    for u,v in edges :
        graph[u].add(v)
        graph[v].add(u)
    ```
- 장점 
    - 불필요한 간선 표현에 소모되는 메모리 비용 감소
    - 간선의 존재 유무도 $O(1)$ 시간 복잡도로 처리 가능

### 인접 딕셔너리
- 인접 행렬의 행,열을 모두 Dictionary로 표현하는 방식
- 동적으로 인접한 노드와의 연결을 추가하고, 삭제 가능
    - 메모리 효율적 사용, 크기 변화에 유연함

- 구현
    ``` python
    graph = {}

    for u,v in edges :
        if u not in graph :
            graph[u] ={}
        graph[u][v] = True

        if v not in graph :
            graph[v] = {}
        graph[v][u] = True
    ``` 

### 가중치 그래프
- 간선마다 가중치가 있는 Graph
    - 앞의 4가지 방식을 응용하여 표현 가능
        1. 인접 행렬로 가중치 그래프 표현 : 기존 1로 표시하던 행렬 값을 가중치로 변경
            ``` python
            adj_list = [[] for _ in range(n+1)]

            for u,v,w in weighted_edges:
                adj_list[u].append(v,w)
                adj_list[v].append(u,w)
            ```
        2. 인접 딕셔너리를 이용하여 구현 : 연결 시, True가 아닌 가중치 값 입력
            ``` python
            graph = {}

            for u,v,w in weighted_edges:
                if u not in graph:
                    graph[u] = {}
                graph[u][v] = w
                if v not in graph:
                    graph[v] = {}
                graph[v][u] = w
            ```
            - 쉽게 인접 딕셔너리 사용하는 방식 (collections library의 defaultdict 사용)
                ``` python
                from collections import defaultdict
                graph = defaultdict(lambda: defaultdict(int))
                for u,v,w in weighted_edges:
                    graph[u][v] = w
                    graph[v][u] = w
                ```

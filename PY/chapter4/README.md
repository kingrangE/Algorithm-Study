# Chapter 4 기본 알고리즘
## 4.1 정렬
- 정렬 : 특정 기준에 맞춰 값을 나열하는 과정
- 정렬 방식 별 성능

    | 정렬 방식 | 최고 성능 | 평균 성능 | 최저 성능 |
    |---|---|---|---|
    | 버블 정렬 | $O(N^2)$ | $O(N^2)$ | $O(N^2)$ |
    | 선택 정렬 | $O(N^2)$ | $O(N^2)$ | $O(N^2)$ |
    | 삽입 정렬 | $O(N)$ | $O(N^2)$ | $O(N^2)$ |
    | 셀 정렬 | $O(N)$ | $O(N^{1.5})$ | $O(N^)$ |
    | 힙 정렬 | $O(NlogN)$ | $O(NlogN)$ | $O(NlogN)$ |
    | 병합 정렬 | $O(NlogN)$ | $O(NlogN)$ | $O(NlogN)$
    | 퀵 정렬 | $O(NlogN)$ | $O(NlogN)$ | $O(N^2)$ |

    - 힙 정렬보다 퀵 정렬의 최저 성능이 더 좋지 않음
        - but, 실제로는 퀵 정렬이 가장 성능이 좋음
            - 분할 방식을 사용하여 실제 실행 시간의 성능을 높이는 것이 가능하기 때문
    - 구현 난이도
        - EASY : 버블, 선택, 삽입, 셀
        - HARD : 힙, 병합, 퀵, 기수
        - 이렇게 성능과 구현 난이도가 다르기에 다양한 부분을 고려하여 데이터 형태에 맞춰 최적화를 통한 튜닝이 필요할 수 있음

### 선택 정렬
- 이름처럼 왼->오 방향으로 이동하며, 가장 작은 값을 선택하여 위치를 옮기는 알고리즘

- 동작 방식
    - EX) [2,1,0,4]를 선택 정렬로 오름차순 정렬하는 상황
        1. i를 포함한 모든 원소중 가장 작은 값을 찾음 (0)
        2. 위에서 찾은 값과 i 번째 원소와 자리 교환
        3. i를 1증가
        4. 1~3과정 반복
- 구현
    ``` python
    arr = [2,1,0,4]
    for i in range(len(arr)):
        min_value = arr[i]
        for j in range(i,len(arr)):
            if min_value > arr[j]:
                min_value = arr[j]
        temp_index = arr.index(min_value)
        arr[i], arr[temp_index] = arr[temp_index], arr[i]
    print(arr)
    ```
### 버블 정렬
- 인접한 두 원소의 크기를 비교해 큰 값을 뒤에 배치하면서 정렬하는 알고리즘

- 동작 방식
    - EX, [2,1,0,4]를 오름차순 정렬
        1. 0~n-1까지 i와 i+1 원소 비교, 큰 값을 뒤로 옮김
        2. 0~n-2까지 i와 i+1 원소 비교, 큰 값을 뒤로 옮김
        3. 1~2한 것처럼 계속 끝단을 줄여가며 반복
- 구현
     ``` python
     arr = [2,1,0,4]
     for i in range(len(arr)):
        for j in range(1,len(arr)-i):
            if arr[i] < arr[j] :
                arr[i],arr[j] = arr[j],arr[i]
     ```

### 병합 정렬
- 분할 정복 방식으로 배열을 정렬하는 방식
    - 분할 정복 방식 
        - 리스트를 분할하여 정렬하는 방식
            1. 가장 작은 단위인 1로 인접한 두 원소를 정렬
            2. 단위를 2,4...로 키우면서 인접한 원소를 정렬
- 동작 방식
    - [2,1,0,4,5,3,7,2]를 오름차순으로 정렬
        1. 먼저 1개 단위로 인접한 원소끼리 비교해 큰 값은 뒤로, 작은 값은 앞으로 정렬
            - [2,1] [0,4] [5,3] [7,2] 비교
            - 1,2,0,4,3,5,2,7이 된다.
        2. 위에서 묶은 것을 두 개씩 묶어 다시 비교함
            - [1,2] [0,4]일때, 앞의 첫번째(1)와 뒤의 첫 번째(0)를 비교 0이 작으므로 결과 리스트에 0 추가
            - 앞의 첫 번째(1)와 뒤의 두 번째(4)를 비교 1이 작으므로 결과 리스트에 1 추가
            - 앞의 두 번째(2)와 뒤의 두 번째(4)를 비교 2가 작으므로 결과 리스트에 2,4 순서대로 추가
            - 뒷 부분도 마찬가지로 반복
        3. 이렇게 정렬된 결과 리스트를 이제 4개씩 묶어 2번처럼 비교
            - [0,1,2,4][2,3,5,7]
        4. 계속 2배수씩 키워가며 2번처럼 비교를 반복

- 구현
    ``` python
    arr = [2,1,0,4,5,3,7,2]
    
    def merge_sort(nums):
        if len(nums) < 2 :
            return nums
        m = len(nums) // 2
        left = merge_sort(nums[:m])
        right = merge_sort(nums[m:])
        merged_nums = []
        lpos = 0 
        rpos = 0
        while lpos < len(left) or rpos < len(right) :
            if lpos < len(left) and rpos < len(right):
                if left[lpos] > right[rpos] :
                    merged_nums.append(right[rpos])
                    rpos += 1
                else :
                    merged_nums.append(left[lpos])
                    lpos += 1
            elif lpos<len(left):
                merged_nums.append(left[lpos])
                lpos += 1
            else :
                merged_nums.append(right[rpos])
                rpos += 1
        return merged_nums

    print(merge_sort(arr))
    ```

### 퀵 정렬
- 병합 정렬처럼 분할 방식으로 정렬하는 알고리즘
    - 차이점 : 크기 단위로 리스트를 나누는 것이 아닌, 특정 위치의 원소를 중심점으로 삼아 정렬
- 동작 방식
    - EX, [2,4,1,5,3] 오름차순 정렬
        1. 중심점을 잡기 (중심 index로 잡아도 되고, 무작위로 잡아도 된다.)
            - 맨 오른쪽 끝을 잡았다 가정 (index: len(arr)-1)
        2. 중심점을 잡았다면, 가장 왼쪽의 원소를 left가 가리키도록, 중심점을 제외한 가장 오른쪽 원소를 right가 가리키도록 한다.
            - `left : 2`, `right : 5` ,`중심점 : 3`
        3. left가 가리키는 위치의 값이 중심점 값보다 작으면 포인터를 오른쪽으로 이동시켜 left왼쪽에는 중심점보다 작은 값이 오도록 함
            - right는 반대로 중심점보다 right의 오른쪽에는 중심점보다 큰 값이 오도록함
            - left(2) < 중심점(3) -> left를 4로 이동
            - right(5) > 중심점(3) -> right를 1로 이동
        4. left > right가 된다면 둘이 값을 교환 
            - left(4) > right(1) 이므로 교환
            - [2,1,4,5,3]이 된다.
        5. 중심점과 left값을 교환 -> left 기준 왼쪽에는 3보다 작은 값 오른쪽에는 3보다 큰 값 위치
            - [2,1,3,5,4]
        6. 이제 기존 피봇이었던 3을 기준으로 [2,1] [5,4]를 나눠 각각 1~5 적용
        7. left를 중심으로 왼쪽과 오른쪽 원소가 하나씩 존재할 때까지 진행

- 구현
    ``` python
    def partition(nums, base, n):
        pivot = nums[base + n - 1]
        l = 0
        r = n - 2
        
        while l <= r:  
            while l < n - 1 and nums[base + l] < pivot:
                l += 1
            while r >= 0 and nums[base + r] > pivot:
                r -= 1
                
            if l <= r:
                nums[base + l], nums[base + r] = nums[base + r], nums[base + l]
                l += 1  
                r -= 1  
        
        nums[base + n - 1], nums[base + l] = nums[base + l], nums[base + n - 1]
        return l

    def quick_sort(nums, base, n):
        if n <= 1:
            return
        m = partition(nums, base, n)
        quick_sort(nums, base, m)
        quick_sort(nums, base + m + 1, n - m - 1) 

    # 테스트
    nums = [7, 5, 2, 1, 4]
    quick_sort(nums, 0, len(nums))
    print(nums)  # [1, 2, 4, 5, 7]

    ```
## 4.2 Graph Algorithm
- DFS, BFS를 시작으로 최단 경로 구하기, 모든 노드를 최소 비용으로 연결하기, 순서를 보장하는 위상 정렬 등을 살펴볼 예정

### BFS(breadth first search)
- 모든 노드를 방문하기 위해 **가까운 주변 노드부터 시작해 점차 반경을 넓혀가면서 탐색**할 때 사용
    - 시작점과 **인접한 노드부터 방문하고 깊이를 증가시키며 방문을 확장**하는 방법

- 동작 방식
    - 가장 먼저 탐색해야 하는 노드가 1번이라 가정
    1. 1번 노드에서 방문할 수 있는 노드 파악 -> 이 노드들을 큐에 저장
    2. 큐에서 노드 1개를 꺼내, 꺼낸 노드에서 방문할 수 있는 노드를 큐에 넣음
    3. 큐가 빌 때까지 2번을 반복

- 구현
    ``` python   
    def bfs(graph,u) :
        q = collections.deque([u]) #시작 노드 u
        visited = {u}
        while q :
            u = q.popleft()
            if u not in graph:
                continue
            for v in graph[u] :
                if v in visited :
                    continue
                visited.add(v)
                q.append(v)
    ```

- 성능 분석
    1. 큐에 값을 넣고 빼는 데는 $O(1)$ 시간 소요 -> $V$ 개 있다면 $O(V)$ 시간 소요
    2. 각 노드는 인접 노드를 순회하고 이 과정에서 모든 연결선이 한 번씩 처리되기에 연결선의 개수만큼 시간 소요
        - $E$개 있다면 $O(E)$ 시간 소요
    3. 따라서 BFS에 소모되는 전체 시간은 $O(V+E)$이다.

### DFS (Depth First Search)
- 성능이나 동작 방식이 BFS와 거의 동일
    - 차이점 : 주변 노드부터 방문하는 것이 아니라 **먼저 방문한 노드의 최말단 노드까지 모두 방문한 다음 주변 노드 방문** + **스택** 사용
        - 즉 주변 노드를 방문하지 않고 처음 시작한 노드에서 도달 가능한 마지막 노드까지 방문을 유지해야 할 값이 있을 때, DFS 사용

- 동작 방식
    1. 먼저 최초 방문하는 노드 1번 노드 가정
    2. 1번 주위에 방문하지 않은 노드를 Stack에 push (2,3,4노드)
    3. 스택은 LIFO이므로 4를 먼저 pop
    4. 4의 주위에 있는 노드를 Stack에 push
    5. Stack이 빌 때까지 3-4 반복
- 구현
    ```python
    def dfs_while(graph,u):
        q = [u]
        visited = {u}
        while q:
            u = q.pop()
            print('visited =',u)

            if u not in graph : 
                continue
            for v in graph[u]:
                if v in visited :
                    continue
                visited.add(v)
                q.append(v)
    ```
    ``` python
    def dfs_recursive(graph,visited,u):
        visited.add(u)
        print('visited =',u)
        if u not in graph:
            return
        for v in graph[u]:
            if v in visited :
                continue
            dfs(graph,visited,v)  
    ```

### 우선 순위 탐색
- 알고리즘 문제의 흔한 유형 : **최단 경로 찾기**
    - 그래프의 시작 정점부터 종료 정점까지 **최소 비용으로 연결할 수 있는 간선의 집합 경로 찾기**
- 대표적 경로탐색 알고리즘
    1. A-star 알고리즘
    2. 다익스트라
    3. 플로이드-워셜
    - 이러한 알고리즘의 기본이 우선순위 탐색

- 동작 방식
    - 힙큐를 이용해서 시작 노드에서 주변 노드로 가는 거리와 함께 주변 노드를 힙큐에 추가
        - EX) 시작 노드(A)와 연결된 노드(B,C)가 각각 2,4의 거리를 갖는다면, (A,B,2)(A,C,4)를 힙큐에 추가
            - 이를 통해서 가장 가까운 것부터 출력할 수 있도록 함
    - 이 과정을 모든 노드에 대해 반복하여 구하기 
- 구현
    ``` python
    def prio_first_search(start_name, edges):
        g = collections.defaultdict(lambda: collections.defaultdict(int))
        for u,v,w in edges:
            g[u][v] = w
            g[v][u] = w
        q = [(0,start_name,start_name)] #최초 방문 노드
        visited = set() 
        edges = []
        while q : # 최소 비용 경로를 방문하고 edge에 저장 반복
            dist,src,u = heapq.heappop(q)
            if u in visited:
                continue
            visited.add(u)
            edges.append((src,u))
            for v,w in g[u].items():
                heapq.heappush(q,(dist+w,u,v))
        return edges # 방복이 완료된 후, 방문에 사용했던 간선 저장하는 edge반환
    ```

### 다익스트라
- 가중치가 있는 그래프의 한 정점 -> 다른 정점까지 가는데 필요한 **최소 비용**을 산출하는 알고리즘
    - 우선순위 탐색 알고리즘과 유사
- 특징
    1. 단일 출발점 최단 경로 : 주어진 출발점에서 다른 모든 노드까지의 최단 경로를 구함
    2. 비음수 가중치 : 간선의 가중치가 음수가 아닌 경우에만 적용 가능
    3. Greedy Approach : 매 단계에서 가장 최단 거리를 갖는 노드를 선택하여 경로 확장
- 동작 원리
    1. 초기화 : 출발 노드의 거리 0 설정, 다른 노드와의 거리 inf설정
    2. 방문하지 않은 노드 중 최단 거리를 가진 노드 선택 : 현재 노드에서 인접한 모든 노드의 거리 update
    3. 거리를 업데이트 : 현재 노드와 인접한 노드간의 거리를 계산하여 더 짧은 경로가 발견되면 거리 update
    4. 반복 : 모든 노드를 방문할 때까지 2,3번 단계 반복
- 활용
    1. 최단 경로 문제
    2. Network Routing
    3. 교통 최적화
- 구현
    ``` python
    import heapq

    def dijkstra(graph, start):
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        priority_queue = [(0, start)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances

    graph = {
        'A': {'B': 1, 'C': 2},
        'B': {'A': 1, 'E': 2, 'D': 6},
        'C': {'A': 2, 'E': 3, 'F': 8},
        'D': {'B': 6, 'E': 1},
        'E': {'B': 2, 'C': 3, 'D': 1, 'F': 7},
        'F': {'C': 8, 'E': 7}
    }

    print(dijkstra(graph, 'A'))
    ```

### 위상 정렬
- 비순환 방향 그래프에만 적용할 수 있다. (ex, 대학교 선수 과목)

- 동작 방식
    1. 진입 차수가 0인 노드들을 queue에 넣음 (진입 차수 : 간선의 수)
    2. queue에서 node를 하나씩 빼며 연결되어 있는 node의 진입 차수를 감소시킴
    3. 1-2 반복

- 구현
    ```python
    def topology_sort(indegree, graph):
        result = []
        queue = deque()
        for i in range(1,n+1):
            if indegree[i] == 0 : #진입차수 0이면
                queue.append(i)
        while queue :
            current = queue.popleft()
            result.append(current)
            for i in graph[current]:
                indegree[i] -=1
                if indegree[i] == 0 :
                    queue.append(i)
        for i in result :
            print(i, end=" ")

    topology_sort()
    ```
## 4.3 문자열 검색
- 주어진 문자 배열에서 원하는 패턴의 배열을 찾는 것
    - 가장 흔한 방법 : 문자열의 길이가 M이고, 찾고자 하는 substring 길이가 N일 때, O(MN)의 시간복잡도로 찾는 것
        - BUT, KMP알고리즘과 rabin-karp 알고리즘으로 이를 O(N)의 시간복잡도로 수행할 수 있게 할 수 있다.
### 가장 흔한 방법
```python
def common_search(full_text,sub_text):
    for i in range(len(full_text)):
        flag = True
        for j in range(len(sub_text)):
            if full_text[i+j] != sub_text[j] :
                flag = False
        if flag : 
            print(full_text[i:i+len(sub_text)])
```
### rabin-karp 알고리즘
- Full text에서 sub text를 O(N)의 시간 복잡도로 찾아내는 알고리즘
- Idea  
    - 찾고자 하는 문자열 길이에 해당하는 문자를 sliding window에 넣고, 각 문자를 특정 hash 값으로 변경
    - 이 값의 총합으로 윈도우에서 해당하는 문자를 식별하는 방식

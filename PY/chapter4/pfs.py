import collections
import heapq

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

edges_input = [
        ('A', 'B', 1),
        ('A', 'C', 4),
        ('A', 'D', 5),
        ('B', 'D', 2),
        ('C', 'D', 3),
    ]
result = prio_first_search('A', edges_input)
print(result)
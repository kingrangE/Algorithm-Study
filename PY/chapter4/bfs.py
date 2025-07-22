import collections

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
                print(v)
                visited.add(v)
                q.append(v)

g = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B'],
    'D': ['B']
}

print(bfs(g,u='A'))
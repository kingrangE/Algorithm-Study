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

def dfs_recursive(graph,visited,u):
        visited.add(u)
        print('visited =',u)
        if u not in graph:
            return
        for v in graph[u]:
            if v in visited :
                continue
            dfs_recursive(graph,visited,v)  

g = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B'],
    'D': ['B']
}

print(dfs_while(g,u='C'))
print(dfs_recursive(g,visited=set(),u='C'))
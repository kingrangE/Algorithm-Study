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

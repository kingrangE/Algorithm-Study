arr = [2,1,0,4]
for i in range(len(arr)):
    min_value = arr[i]
    for j in range(i,len(arr)):
        if min_value > arr[j]:
            min_value = arr[j]
    temp_index = arr.index(min_value)
    arr[i], arr[temp_index] = arr[temp_index], arr[i]
print(arr)
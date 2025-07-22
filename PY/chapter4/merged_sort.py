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
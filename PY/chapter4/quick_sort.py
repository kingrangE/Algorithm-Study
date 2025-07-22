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

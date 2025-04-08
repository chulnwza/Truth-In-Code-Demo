def sum_of_unique(nums):
    """
    รับลิสต์ของตัวเลข แล้วคืนผลรวมของตัวเลขที่ไม่ซ้ำกันเท่านั้น
    เช่น [1,2,3,2] -> return 1 + 3 = 4
    """
    frequency = {}
    
    for num in nums:
        frequency[num] = frequency.get(num, 0) + 1

    total = 0
    for num, count in frequency.items():
        if count == 1:
            total += num

    return total

# ทดสอบฟังก์ชัน
if __name__ == "__main__":
    test_list = [4, 5, 7, 5, 4, 8]
    result = sum_of_unique(test_list)
    print("ผลรวมของตัวเลขที่ไม่ซ้ำคือ:", result)

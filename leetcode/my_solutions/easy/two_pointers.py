from typing import List

# =========================== START OF QUESTION 349 ==================================
# element must appear only once
class TwoArrayIntersectionSolution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1_length, nums2_length = len(nums1), len(nums2)
        if nums2_length - nums1_length:
            shorter_arr, longer_arr = nums1, nums2  
        else:
            shorter_arr, longer_arr = nums2, nums1
        intersect = []

        for item in shorter_arr:
            if item in longer_arr and item not in intersect:
                intersect.append(item)
        return intersect
    
# =========================== END OF QUESTION 349 ==================================

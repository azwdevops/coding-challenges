def minMaxSum(arr):
    arr.sort()
    minSum = sum(arr[:len(arr)-1])
    maxSum = sum(arr[1:])
    print(minSum, maxSum)


# minMaxSum([1, 8, 6, 2, 3, 2, 5, 4])


def timeConversions(s):
    hours, minutes, seconds, status = s[:2], s[3:5], s[6:8], s[-2:]
    print(hours, minutes, seconds, status)
    rest = f':{minutes}:{seconds}'
    if int(hours) == 12:
        if status == 'AM':
            return f'00{rest}'
        elif status == 'PM':
            return f'12{rest}'
    else:
        if status == 'AM':
            return f'{hours}{rest}'
        elif status == 'PM':
            return f'{int(hours)+12}{rest}'


# print(timeConversions('12:01:52PM'))
# print(timeConversions('07:01:52AM'))
# print(timeConversions('12:45:54PM'))

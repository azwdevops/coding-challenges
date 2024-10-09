def reverse_str(str):
    new_str = ""
    for i in range(len(str) - 1, -1, -1):
        new_str += str[i]

    return new_str


def reverse_str1(str):
    return "".join(reversed(str))


def reverse_str2(str):
    return str[::-1]


# print(reverse_str('abcdef'))
# print(reverse_str1('abcdef'))
# print(reverse_str2('abcdef'))


def add_nums(num1, num2):
    while num2 != 0:
        data = num1 & num2
        num1 = num1 ^ num2
        num2 = data << 1
    return num1


# print(add_nums(20, 52))
def fib(n):
    table = [0] * (n + 1)
    table[0] = 1
    table[1] = 1
    for i in range(2, n + 1):
        table[i] = table[i - 1] + table[i - 2]
    return table


# print(fib(10))


def is_prime(num):
    for i in range(2, num):
        if num % i == 0:
            return False
    return True


# print(is_prime(1))
# print(is_prime(6))


def bubble_sort(nums):
    n = len(nums)
    for i in range(n):
        swapped = False

        for j in range(n - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                swapped = True

        if not swapped:
            break

    return nums



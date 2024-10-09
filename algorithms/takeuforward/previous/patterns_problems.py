def print_pattern_7(n):
    for i in range(n):
        # space
        for j in range(n - i - 1):
            print(" ", end="")
        # stars
        for j in range(2 * i + 1):
            print("*", end="")
        # space
        for j in range(n - i - 1):
            print(" ", end="")
        print("")


# print_pattern_7(7)


def print_pattern_8(n):
    for i in range(n):
        for j in range(i):
            print(" ", end="")
        for j in range(2 * n - (2 * i + 1)):
            print("*", end="")
        for j in range(i):
            print(" ", end="")
        print("")


# print_pattern_8(5)


def print_pattern_9(n):
    print_pattern_7(n)
    print_pattern_8(n)


# print_pattern_9(5)


def print_pattern_10(n):
    for i in range(2 * n):
        stars = i
        if i > n:
            stars = 2 * n - i
        for j in range(stars):
            print("*", end="")
        print("")


# print_pattern_10(5)


def print_pattern_11(n):
    for i in range(n):
        if i % 2 == 0:
            start = 1
        else:
            start = 0
        for j in range(i + 1):
            print(start, end="")
            start = 1 - start
        print("")


# print_pattern_11(5)


def print_pattern_12(n):
    space = 2 * (n - 1)
    for i in range(1, n + 1):
        # numbers
        for j in range(1, i + 1):
            print(j, end="")
        # spaces
        for j in range(1, space + 1):
            print(" ", end="")
        # numbers
        for j in range(i, 0, -1):
            print(j, end="")
        print("")
        space -= 2


# print_pattern_12(5)


def print_pattern_13(n):
    num = 1
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            print(num, end=" ")
            num += 1
        print("")


# print_pattern_13(5)


def print_pattern_14(n):
    for i in range(n):
        char_index = ord("A")
        for i in range(i + 1):
            print(chr(char_index + i), end="")
        print("")


# print_pattern_14(5)


def print_pattern_15(n):
    for i in range(n, 0, -1):
        char_index = ord("A")
        for i in range(i):
            print(chr(char_index + i), end="")
        print("")


# print_pattern_15(5)


def print_pattern_16(n):
    char_index = ord("A")
    for i in range(1, n + 1):
        for j in range(i):
            print(chr(char_index), end="")
        print("")
        char_index += 1


# print_pattern_16(5)


def print_pattern_17(n):
    for i in range(n):
        # spaces
        for j in range(n - i - 1):
            print(" ", end="")
        # characters
        char_index = ord("A")
        breakpoint = (2 * i + 1) / 2
        for j in range(1, 2 * i + 2):
            print(chr(char_index), end="")
            if j <= breakpoint:
                char_index += 1
            else:
                char_index -= 1

        # spaces
        for j in range(n - i - 1):
            print(" ", end="")

        print("")


print_pattern_17(5)

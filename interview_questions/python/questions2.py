x = [1, 2, 3, 4]
y = [i**2 for i in x]
# print(y)


def func(x=[]):
    x.append(1)
    return x


# print(func())
# print(func())


def foo(bar=[]):
    bar.append("baz")
    return bar


# print(foo())
# print(foo())


# a = [1, 2, 3]
# b = a
# a.append(4)
# print(b)
def foo():
    return "bar"


# print(foo() or "baz")
lst = [1, 2, 3, 4, 5]


# print(lst[::-2])
def greet(name):
    return f"Hello, {name}"


# print(greet("World"))
# print(greet(""))

# name = "John"
# age = 30
# print("My name is " + name + ", and I am " + age + " years old.")


# numbers = [1, 2, 3, 4, 5]
# squares = map(lambda x: x**2, numbers)
# print(list(squares))

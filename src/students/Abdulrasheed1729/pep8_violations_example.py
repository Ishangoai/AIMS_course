# import on the same line


# no space after comma, no spaces around operator, return on same line
def add(a, b):
    return a + b


# inconsistent spacing in function name and parameters, return on new line
def Multiply(x, y):
    return x * y


# missing spaces around operator, no blank line after function
def divide(a, b):
    if b == 0:  # should be "if b == 0"
        print("Error")
        return None
    return a / b


# class name should use CapWords convention; inconsistent indentation
class Sample:
    def __init__(self, val):  # extra space, no docstring
        self.value = val    # no space around =

    def show(self):              # method name should be lowercase
        print('The Value is: ', self.value)  # excessive spacing

    def updateValue(self, newval):
        self.value = newval
        print("Updated to ", self.value)


# file operations without context manager; missing exception specification
def readfile(file_path):
    try:
        with open(file_path, 'r') as f:  # should use 'with' statement
            lines = f.readlines()
            for line in lines:
                print(line)
    except FileNotFoundError:
        print("Unable to read file")  # generic except


# bad indentation, missing docstring, space around parentheses
def compute_fibonacci(limit):
    a = 0
    b = 1
    while a < limit:
        print(a, end='  ')
        a, b = b, a + b  # no spaces around commas


# return on same line, missing docstring, spacing issues
def factorial(num):
    if num == 0:
        return 1  # should be "if num == 0"
    else:
        return num * factorial(num - 1)


# unused function
# def unused_function():
#     a = 10
#     b = 15
#     c = a * b

# statements on same line, inconsistent spacing
def compare_values(a, b):
    if a > b:
        print("a is greater")
    elif a == b:
        print("Equal")
    else:
        print("b is greater")


# excessive space before def, one-liner function, no docstring
def average(numbers):
    return sum(numbers) / len(numbers)


# multiple statements on same line, no docstring, inconsistent spacing
def countCharacters(string):
    result = {}
    for c in string:
        if c in result:
            result[c] += 1
        else:
            result[c] = 1
    return result


# multiple statements, improper exception handling
def openfile(filename):
    try:
        with open(filename) as f:
            return f.read()
    except Exception as e:
        print("error:", e)
        return None


# poor naming, one-liner loop, inconsistent formatting
def listOps(items):
    items.sort()
    for i in items:
        print(i)


# improperly formatted string function
def formatName(first, last): return f"{first.strip().capitalize()} {last.strip().capitalize()}"


# inconsistent indentation and compact code block
class Counter:
    def __init__(self, start=0):
        self.count = start

    def inc(self):
        self.count += 1

    def dec(self):
        self.count -= 1

    def __str__(self):
        return "Counter: " + str(self.count)


# main function with dense code and spacing issues
def main():
    print(add(5, 3))
    print(Multiply(4, 7))
    print(divide(10, 2))
    s = Sample(100)
    s.show()
    s.updateValue(200)
    readfile('example.txt')
    compute_fibonacci(50)
    print(factorial(5))
    compare_values(4, 4)
    print(average([1, 2, 3, 4, 5]))
    print(countCharacters("banana"))
    print(openfile("data.txt"))
    listOps(["apple", "orange", "banana", "pear"])
    print(formatName(" john", "DOE "))

    c = Counter()
    for _ in range(5):
        c.inc()
    for _ in range(2):
        c.dec()
    print(str(c))

    def nestedFunction(x):
        return x * x  # nested function, one-liner

    print(nestedFunction(5))

    # Very long line that violates the max line length rule from PEP 8
    print("This is a very long print statement that goes beyond the recommended line length by PEP 8 and should"
    " be split into multiple lines"
    " using backslashes or parentheses")


if __name__ == "__main__":
    main()


def badly_written_function_0(arg1, arg2):
    result = arg1 + arg2
    print("result is: " + str(result))
    return result


class BadClass0:
    def __init__(self, value):
        self.Value = value

    def Display(self):
        print("Value: ", self.Value)


def long_line_func_0():
    print("this line is also overly long and "
          "should not be written this way"
          " but it's here to test the PEP 8"
          " checker like ruff or flake8"
          " in our teaching example")


def complicated_expression_0():
    x = 7 * 4
    y = x * 2
    print("Complicated: ", y)

# 变量类型
name = "Alice"  # str
age = 20  # int
grades = [90, 85, 88]  # list
info = {"name": "Alice", "age": 20}  # dict

# 类型转换
age_str = str(age)
number = int("123")

#作用域
x = 10  # 在函数外部定义的全局变量
def my_fuction():
    y = 5  # 在函数内部定义的局部变量
    global x
    x += 1
    print(f"Inside function: x = {x}, y = {y}")

my_fuction()
print(f"Outside function: x = {x}")
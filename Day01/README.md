# Day01

1. ### 环境准备

   - win + R并输入cmd，在控制台进行操作；

   - 使用**“python --version”**来查看python版本

   - 若在控制台显示当前python版本，说明路径信息配置成功，否则需要进行环境变量的配置。

     

2. ### Python编译入门教学

   #### 2.1 Python变量类型及其作用域

   - python基本的变量类型：int，float，double，bool，list；

   - 作用域：全局，局部（使用global和nonlocal关键字来定义）；

   - 类型转换：使用int()，str()来进行强制类型转换。

     ```c++
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
     ```

     

   #### 2.2 运算符

   - 算数运算符：加减乘除，取余；

   - 比较运算符：=，!=，<，>；

   - 逻辑运算符：and，or，not；

     ```c++
     # 算术运算
     a = 10
     b = 3
     
     print(a + b)
     print(a // b)
     print(a ** b)
     
     x = True
     y = False
     print(x and y)
     print(x or y)
     
     #比较运算，a>b返回True，否则返回False
     print(a > b)
     ```

     

   #### 2.3 条件语句，循环语句，异常

   - 条件语句：if，elif，else；

   - 循环语句：for，while，continue，break；

   - 异常处理：try，except，finally。

     ```c++
     # 条件语句
     score = 85
     if score >= 90:
         print("A")
     elif score >= 60:
         print("Pass")
     else:
         print("Fail")
     
     # 循环语句
     for i in range(5):
         if i == 3:
             continue
         print(i)
     
     # 异常处理
     try:
         num = int(input("Enter a number: "))
         print(100 / num)
     except ZeroDivisionError:
         print("You can't divide by zero")
     except ValueError:
         print("Invalid nput")
     finally:
         print("Execution completed.")
     ```

     

   #### 2.4 函数的定义，参数，匿名函数，高阶函数的使用

   - 函数的定义使用**”def“**关键字，并且可以使用**默认参数**和**可变参数(args, *kwargs)**两种定义方式；

   - 匿名函数：lambda；

   - 高阶函数：接受函数作为参数或者返回函数。

     ```c++
     # 函数定义
     def greet(name, greeting = "Hello"):
         return f"{greeting} {name}!"
     
     print(greet("Alice"))
     print(greet("Bob", "Hi"))
     
     # 可变参数
     def sum_numbers(*args):
         return sum(args)
     print(sum_numbers(1, 2, 3, 4))
     
     # 匿名函数
     double = lambda x : x * 2
     print(double(5))
     
     # 高阶函数
     def apply_func(func, value):
         return func(value)
     print(apply_func(lambda x : x ** 2, 4))
     ```

     

   #### 2.5 包和模块：定义，导入，使用，第三方模块管理

   - 使用**from .. import ...** 来导入模块；

   - 创建模块：新建一个.py后缀的文件；

   - 包：包含 _ _init__.py的文件夹；

   - 第三方模块：如requests，numpy。

     ```c++
     # 创建模块 mymodule.py
     # mymodule.py
     def say_hello(name):
         return "Hello from module!"
     
     # 主程序
     import mymodule
     print(mymodule.say_hello())
     
     # 导入第三方模块
     import requests
     response = requests.get('https ://api.github.com')
     print(response.status_code)
     
     from mypackage import mymodule
     ```

     

   #### 2.6 类和对象

   - 类的定义：使用class关键字，类中包含各种属性和方法；

   - 继承，多态，封装（类的三大特性）；

   - 实例化对象。

     ```c++
     # 定义类
     class Student:
         def __init__(self, name, age):
             self.name = name
             self.age = age
     
         def introduce(self):
             return f"I am {self.name}, I am {self.age} years old!"
     
     # 继承
     class GradStudent(Student):
         def __init__(self, name, age, major):
             super().__init__(name, age)
             self.major = major
     
         def introduce(self):
             return f"I am {self.name}, a {self.major} student."
     
     # 使用
     student = Student("Alice", 20)
     grad = GradStudent("Bob", 22, "CS")
     print(student.introduce())
     print(grad.introduce())
     ```

     

   #### 2.7 装饰器

   - 装饰器的本质：高阶函数，可以接受函数并返回新的函数；

   - 使用@语法来进行装饰器的使用；

   - 带参数的装饰器使用。

     ```c++
     # 简单装饰器
     def my_decorator(func):
         def wrapper():
             print("Befor function")
             func()
             print("After function")
         return wrapper
     
     @my_decorator
     def say_hello():
         print("Hello")
     
     say_hello()
     
     # 带参数的装饰器
     def repeat(n):
         def decorator(func):
             def wrapper(*args, **kwargs):
                 for i in range(n):
                     func(*args, **kwargs)
             return wrapper
         return decorator
     
     @repeat(3)
     def greet(name):
         print(f"Hi, {name}!")
     
     greet("Alice")
     ```

     

   #### 2.8 文件操作

   - 使用**open()**，**read()**，**write()**来分别进行文件的打开与读写操作；

   - 上下文管理器：with语句；

   - 处理csv，json文件。

     ```c++
     # 写文件
     with open("example.txt", "w") as f:
         f.write("Hello Python!\n")
     
     # 读取文件
     with open("example.txt", "r") as f:
         contents = f.read()
         print(contents)
     
     # 处理csv
     import csv
     with open("data.csv", "w", newline="") as f:
         writer = csv.writer(f)
         writer.writerow(["Name", "Age"])
         writer.writerow(["Alice", 20])
     
     
     ```

     

3. ### 在Pycharm使用git进行仓库管理

   - 仓库初始化：git init
   - 添加到本地暂存区：git add .
   - 提交到本地仓库：git commit -m ""
   - 推送到远程仓库：git push origin main
   - 设置全局名称：git config -global user.name ""
   - 设置全局提交邮箱：git config _global user.email
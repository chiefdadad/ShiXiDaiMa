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


# 训练自己的数据集

1. 新建一个dataset文件夹用于存放数据集处理代码；

   - 添加一个**deal_with_datasets.py**的文件，用于将数据集划分为train以及val两部分，且二者之比为7 ： 3；

     ```python
     import os
     import shutil
     from sklearn.model_selection import train_test_split
     import random
     
     # 设置随机种子以确保可重复性
     random.seed(42)
     
     # 数据集路径
     dataset_dir = r'D:\ShiXiDaiMa\Day03\Images' 
     train_dir = r'D:\ShiXiDaiMa\Day03\image2\train'  # 训练集输出路径
     val_dir = r'D:\ShiXiDaiMa\Day03\image2\val'  # 验证集输出路径
     
     # 划分比例
     train_ratio = 0.7
     
     # 创建训练集和验证集目录
     os.makedirs(train_dir, exist_ok=True)
     os.makedirs(val_dir, exist_ok=True)
     
     # 遍历每个类别文件夹
     for class_name in os.listdir(dataset_dir):
         if class_name not in ["train", "val"]:
             class_path = os.path.join(dataset_dir, class_name)
     
             # 获取该类别下的所有图片
             images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
             # 确保图片路径包含类别文件夹
             images = [os.path.join(class_name, img) for img in images]
     
             # 划分训练集和验证集
             train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)
     
             # 创建类别子文件夹
             os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
             os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
     
             # 移动训练集图片
             for img in train_images:
                 src = os.path.join(dataset_dir, img)
                 dst = os.path.join(train_dir, img)
                 shutil.move(src, dst)
     
             # 移动验证集图片
             for img in val_images:
                 src = os.path.join(dataset_dir, img)
                 dst = os.path.join(val_dir, img)
                 shutil.move(src, dst)
     
             shutil.rmtree(class_path)
     ```

     

   - 添加一个**prepare.py**文件，目的是生成两个文本文件分别用于保存数据集train和val中每个图片的路径信息；

     ```python
     ### prepare.py
     
     import os
     
     # 创建保存路径的函数
     def create_txt_file(root_dir, txt_filename):
         with open(txt_filename, 'w') as f:
             for label, category in enumerate(os.listdir(root_dir)):
                 category_path = os.path.join(root_dir, category)
                 if os.path.isdir(category_path):
                     for img_name in os.listdir(category_path):
                         # 使用绝对路径写入文件
                         img_path = os.path.join(category_path, img_name)
                         f.write(f"{img_path} {label}\n")
     
     # 使用原始字符串（r''）处理路径中的空格
     create_txt_file(r'D:\ShiXiDaiMa\Day03\image2\train', 'train.txt')
     create_txt_file(r'D:\ShiXiDaiMa\Day03\image2\val', 'val.txt')
     
     ```

     

2. 在与模型同级的目录中加入数据集加载函数，用于将处理好的数据集传入到模型中；

   ```python
   import os
   from PIL import Image
   from torch.utils import data
   
   class ImageTxtDataset(data.Dataset):
       def __init__(self, txt_path: str, folder_name: str, transform=None):
           self.transform = transform
           self.data_dir = folder_name  # 基础文件夹路径
           self.imgs_path = []
           self.labels = []
           with open(txt_path, 'r') as f:
               lines = f.readlines()
           for line in lines:
               parts = line.strip().split()  # 分割每一行
               img_path = parts[0]  # 图像路径是第一个元素
               label = int(parts[-1])  # 标签是最后一个元素
               self.labels.append(label)
               # 确保图像路径是完整的绝对路径
               full_img_path = os.path.join(self.data_dir, img_path)
               self.imgs_path.append(full_img_path)
   
       def __len__(self):
           return len(self.imgs_path)
   
       def __getitem__(self, i):
           path, label = self.imgs_path[i], self.labels[i]
           image = Image.open(path).convert("RGB")
           if self.transform is not None:
               image = self.transform(image)
           return image, label
   ```

   

3. 运行程序使用自己处理的数据集进行模型训练。

   ```python
   import time
   import torch.optim
   import torchvision
   from torch import nn
   from torch.utils.data import DataLoader
   from torch.utils.tensorboard import SummaryWriter
   from model import AlexNet
   from dataset import ImageTxtDataset
   
   # 使用GPU版本进行模型训练
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # 数据增强和归一化
   train_transform = torchvision.transforms.Compose([
       torchvision.transforms.RandomHorizontalFlip(),
       torchvision.transforms.RandomCrop(32, padding=4),
       torchvision.transforms.ToTensor(),
       torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   
   test_transform = torchvision.transforms.Compose([
       torchvision.transforms.ToTensor(),
       torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   
   # 准备数据集
   train_data = ImageTxtDataset(txt_path='D:\\ShiXiDaiMa\\Day03\\train.txt',
                                folder_name='D:\\ShiXiDaiMa\\Day03\\image2\\train',
                                transform=train_transform)
   test_data = ImageTxtDataset(txt_path='D:\\ShiXiDaiMa\\Day03\\val.txt',
                               folder_name='D:\\ShiXiDaiMa\\Day03\\image2\\val',
                               transform=test_transform)
   
   # 数据集长度
   train_data_size = len(train_data)
   test_data_size = len(test_data)
   print(f"训练数据集的长度: {train_data_size}")
   print(f"测试数据集的长度: {test_data_size}")
   
   # 检查标签是否在合法范围内
   max_label = max(train_data.labels)
   print("训练数据中的最大标签:", max_label)
   num_classes = max_label + 1  # 确保 num_classes 至少与最大标签值相等
   
   # 加载数据集
   train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
   test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)
   
   # 创建网络模型
   model = AlexNet(num_classes=num_classes).to(device)  # 将模型移动到GPU
   
   # 创建损失函数
   loss_fn = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到GPU
   
   # 优化器 - 使用Adam优化器
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   # 设置训练网络的一些参数
   total_train_step = 0
   total_test_step = 0
   epoch = 30  # 设置训练次数为30
   
   # 添加tensorboard
   writer = SummaryWriter("./logs_train")
   
   start_time = time.time()
   
   for i in range(epoch):
       print(f"-----第 {i + 1} 轮训练开始-----")
       model.train()
       running_loss = 0.0
       correct = 0
       total = 0
   
       for data in train_loader:
           inputs, labels = data
           inputs, labels = inputs.to(device), labels.to(device)  # 数据移动到GPU
   
           optimizer.zero_grad()
   
           outputs = model(inputs)
           loss = loss_fn(outputs, labels)
           loss.backward()
           optimizer.step()
   
           running_loss += loss.item()
           _, predicted = outputs.max(1)
           total += labels.size(0)
           correct += predicted.eq(labels).sum().item()
   
           total_train_step += 1
           if total_train_step % 100 == 0:
               print(f"第 {total_train_step} 步的训练loss: {loss.item()}")
               writer.add_scalar("train_loss", loss.item(), total_train_step)
   
       # 计算训练准确率
       train_accuracy = 100. * correct / total
       print(f"训练准确率: {train_accuracy:.2f}%")
   
       # 测试阶段
       model.eval()
       test_loss = 0.0
       correct = 0
       total = 0
   
       with torch.no_grad():
           for data in test_loader:
               images, labels = data
               images, labels = images.to(device), labels.to(device)
   
               outputs = model(images)
               loss = loss_fn(outputs, labels)
   
               test_loss += loss.item()
               _, predicted = outputs.max(1)
               total += labels.size(0)
               correct += predicted.eq(labels).sum().item()
   
       test_accuracy = 100. * correct / total
       print(f"测试集上的loss: {test_loss / len(test_loader)}")
       print(f"测试准确率: {test_accuracy:.2f}%")
   
       writer.add_scalar("test_loss", test_loss / len(test_loader), i)
       writer.add_scalar("test_accuracy", test_accuracy, i)
   
       # 保存模型
       torch.save(model.state_dict(), f"model_save/alexnet_{i}.pth")
       print("模型已保存")
   
   writer.close()
   print("训练完成")
   print(f"总训练时间: {time.time() - start_time}秒")
   
   # 十轮训练后：
   # 训练准确率: 26.09%
   # 测试集上的loss: 2.7391061195305415
   # 测试准确率: 29.06%
   # 模型已保存
   # 训练完成
   # 总训练时间: 1712.4348423480988秒
   ```

   

# Day03作业

​		将自己的数据集进行处理，分为train和val两个部分，并且使用加载数据集的函数传入模型进行训练。



**运行结果：**

 十轮训练后：
 训练准确率: 26.09%
 测试集上的loss: 2.7391061195305415
 测试准确率: 29.06%
 模型已保存
 训练完成
 总训练时间: 1712.4348423480988秒
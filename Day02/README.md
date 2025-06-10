# Day02

1. ### 基本概念

   - 训练一定是两次循环；

   - 欠拟合：训练的训练数据集表现不好，且验证表现也不好；

   - 过拟合：训练数据的训练过程表现得很好，但是验证过程表现不好。

     

2. ### 卷积神经网络

   - 卷积过程

     ```python
     import torch
     import torch.nn.functional as F
     input = torch.tensor([[1,2,0,3,1],
                           [0,1,2,3,1],
                           [1,2,1,0,0],
                           [5,2,3,1,1],
                           [2,1,0,1,1]])
     kernel = torch.tensor([[1,2,1],
                            [0,1,0],
                            [2,1,0]])
     
     # 不满足conv2d的尺寸要求
     print(input.shape)
     print(kernel.shape)
     
     # 尺寸变换
     input = torch.reshape(input,(1,1,5,5))
     kernel = torch.reshape(kernel,(1,1,3,3))
     print(input.shape)
     print(kernel.shape)
     
     output = F.conv2d(input=input,weight=kernel,stride=1)
     print(output)
     
     output2 = F.conv2d(input=input,weight=kernel,stride=2)
     print(output2)
     
     # padding 在周围扩展一个像素，默认为0；
     output3 = F.conv2d(input=input,weight=kernel,stride=1,padding=1)
     print(output3)
     ```

     5 * 5的输入数据 3 * 3的卷积核 步长1 填充1

     

   - 图片卷积

     ```python
     import torch
     import torchvision
     from torch.utils.data import DataLoader
     from torch import nn
     from torch.utils.tensorboard import SummaryWriter
     
     dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                            train=False,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)
     dataloader = DataLoader(dataset=dataset,
                             batch_size=64)
     
     
     class CHEN(nn.Module):
         def __init__(self):
             super().__init__()
             self.conv1 = nn.Conv2d(in_channels=3,
                                    out_channels=6,
                                    kernel_size=3,
                                    stride=1,
                                    padding=0)
     
         def forward(self, x):
             x = self.conv1(x)
             return x
     
     
     chen = CHEN()
     print(chen)
     
     writer = SummaryWriter("conv_logs")
     step = 0
     for data in dataloader:
         imgs, targets = data
         output = chen(imgs)
     
         # print(imgs.shape)  # torch.Size([64, 3, 32, 32])
         # print(output.shape)  # torch.Size([64, 6, 30, 30])
         writer.add_images("input", imgs, step)
     
         # torch.Size([64, 6, 30, 30]) ->([**, 3, 30, 30])
         output = torch.reshape(output, (-1, 3, 30, 30))  # -1:会根据后面的值进行调整
         writer.add_images("output", output, step)
         step += 1
     
     定义我们的网络模型
     ```

     

3. ### tensorboard的使用方法

   - 安装tensorboard；

   - 作用：拿到conv_logs里面的文件；

   - 使用tensorboard命令打开；

   - tensorboard --logdir=conv_logs；

   - 点击链接得到界面。

     

4. ### 池化层（本代码展示最大池化，除此之外还有平均池化）

   ```python
   import torch
   import torchvision
   from torch import nn
   from torch.nn import MaxPool2d
   from torch.utils.data import DataLoader
   from torch.utils.tensorboard import SummaryWriter
   #
   dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                          train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
   dataloader = DataLoader(dataset=dataset,
                           batch_size=64)
   
   # # 最大池化没法对long整形进行池化
   # input = torch.tensor([[1,2,0,3,1],
   #                       [0,1,2,3,1],
   #                       [1,2,1,0,0],
   #                       [5,2,3,1,1],
   #                       [2,1,0,1,1]], dtype = torch.float)
   # input =torch.reshape(input,(-1,1,5,5))
   # print(input.shape)
   
   
   class Chen(nn.Module):
       def __init__(self):
           super().__init__()
           self.maxpool_1 = MaxPool2d(kernel_size=3,
                                      ceil_mode=False)
       def forward(self,input):
           output = self.maxpool_1(input)
           return output
   
   chen = Chen()
   
   writer = SummaryWriter("maxpool_logs")
   step = 0
   for data in dataloader:
       imgs, targets = data
       writer.add_images("input",imgs,step)
       output = chen(imgs)
       writer.add_images("ouput",output,step)
       step += 1
   writer.close()
   
   #
   # output = chen(input)
   # print(output)
   ```

   

5. ### Day02练习作业：搭建Alexnet模型并进行训练（要求GPU版本正确率超过0.5）
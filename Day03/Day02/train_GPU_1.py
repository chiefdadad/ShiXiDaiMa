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
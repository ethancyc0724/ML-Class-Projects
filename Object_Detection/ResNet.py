import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision 
import torchvision.transforms as transforms
from PIL import Image


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(batch_size=128, 
                    cifar100_mean=(0.5071, 0.4867, 0.4408), 
                    cifar100_std=(0.2675, 0.2565, 0.2761)):
    # CIFAR-100 官方統計的 mean / std（RGB 三個通道）：
    # mean = (0.5071, 0.4867, 0.4408)
    # std  = (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # 隨機裁切 (data augmentation)
        transforms.RandomHorizontalFlip(),      # 隨機水平翻轉
        transforms.ToTensor(),                  # 轉成 Tensor，範圍 [0,1]
        transforms.Normalize(
            mean=cifar100_mean,
            std=cifar100_std
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # 訓練集要打亂
        num_workers=2,      # 開啟 child process
        pin_memory=True     # 固定記憶體位置
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,      # 測試集不需要打亂
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


def get_resnet():
    from torchvision.models import resnet34

    model = resnet34(weights=None)  # 不用預訓練權重（ImageNet 是 224x224）

    # 原 ResNet-18 第一層：7x7 kernel, stride=2, padding=3 + maxpool
    # 對 32x32 太兇，所以改成 3x3, stride=1 並移除 maxpool
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    model.maxpool = nn.Identity()  # 等價於什麼都不做的 layer

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)

    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # 切換成訓練模式（啟用 Dropout, BatchNorm 的訓練行為）
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)  # outputs 形狀: [batch_size, 100]

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        # 統計訓練過程中的 loss 與 accuracy
        running_loss += loss.item() * images.size(0)  # 累計總 loss
        _, predicted = outputs.max(1)                # 取每一列最大的 logit 的 index 當預測類別
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()  # 切換成eval模式（關閉 Dropout, BatchNorm 用 moving average）
    running_loss = 0.0
    correct = 0
    total = 0

    # 評估時不需要計算梯度，可省記憶體與加速
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def predict(model, img_path, device, cifar100_mean, cifar100_std, classes):
    model.eval()

    inference_transform = transforms.Compose([     
        transforms.Resize((32, 32)),                      
        transforms.ToTensor(),                            
        transforms.Normalize(cifar100_mean, cifar100_std) 
    ])
    img = Image.open(img_path).convert("RGB")
    img = inference_transform(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        pred_class = torch.argmax(outputs, dim=1).item()
        print(img_path, "→", f"{classes[pred_class]}")

    
def main():
    # ==== 超參數設定 ====
    num_epochs = 100
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std  = (0.2675, 0.2565, 0.2761)

    device = get_device()
    print("Using device:", device)

    train_loader, test_loader = get_dataloaders(
    batch_size=batch_size,
    cifar100_mean=cifar100_mean,
    cifar100_std=cifar100_std,
)
    classes = train_loader.dataset.classes
    model = get_resnet()
    model = model.to(device)

    # 損失函數：多類別分類常用 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # 最佳化器：SGD + momentum + weight decay（ResNet 在 CIFAR 常見設定）
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # 學習率衰減：每 10 個 epoch 把 lr 乘上 0.1
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50,75],
        gamma=0.1
    )

    # ==== 訓練迴圈 ====
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    root = "./data/test_object"
    path = [f"{root}/bus.jpg", f"{root}/forest.jpg", f"{root}/dolphin.jpg"]
    
    for img_path in path:
        predict(model, img_path, device, cifar100_mean, cifar100_std, classes)
    
    save_path = "resnet34_cifar100.pth"
    torch.save(model.state_dict(), save_path)
    print("模型已儲存：", save_path)
    
if __name__ == "__main__":
    main()

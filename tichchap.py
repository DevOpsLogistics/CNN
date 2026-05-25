import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# Cấu hình thiết bị (Sử dụng GPU nếu có, ngược lại dùng CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Các siêu tham số (Hyperparameters)
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# Chuẩn bị dữ liệu MNIST
# Transform: Chuyển dữ liệu sang Tensor và chuẩn hóa (Normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Tải tập dữ liệu huấn luyện
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transform,  
                                           download=True)

# Tải tập dữ liệu kiểm thử
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transform)

# DataLoader giúp load dữ liệu theo từng batch
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# Xây dựng Mạng Nơ-ron Tích chập (CNN)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # Lớp Convolution 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Lớp Convolution 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Lớp Fully Connected (Kết nối đầy đủ)
        # Kích thước ảnh sau 2 lớp MaxPool: 28x28 -> 14x14 -> 7x7
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # Flatten (làm phẳng) dữ liệu
        out = self.fc(out)
        return out

# Khởi tạo mô hình
model = ConvNet(num_classes).to(device)

# Định nghĩa Hàm mất mát (Loss function) và Bộ tối ưu hóa (Optimizer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    model_path = 'mnist_cnn_model.pth'
    
    # Kiểm tra xem mô hình đã được huấn luyện và lưu trước đó chưa
    if os.path.exists(model_path):
        print(f"Tìm thấy mô hình đã lưu tại '{model_path}'.")
        print("Đang tải mô hình...")
        # weights_only=True để bảo mật hơn theo khuyến nghị mới của PyTorch
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Tải mô hình thành công. Bỏ qua bước huấn luyện.")
    else:
        # Huấn luyện mô hình
        total_step = len(train_loader)
        print(f"Đang chạy trên thiết bị: {device}")
        print("Bắt đầu huấn luyện mô hình...")

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Lan truyền tiến (Forward pass)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Lan truyền ngược và tối ưu hóa (Backward and optimize)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Bước [{i+1}/{total_step}], Loss: {loss.item():.4f}')
                    
            # Lưu mô hình sau mỗi epoch
            model_filename = f'mnist_cnn_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_filename)
            print(f"Đã lưu mô hình của epoch {epoch+1} vào '{model_filename}'")
            
        # Lưu trọng số của mô hình cuối cùng
        torch.save(model.state_dict(), model_path)
        print(f"Đã lưu trọng số mô hình vào '{model_path}'")

    # Kiểm tra mô hình
    model.eval()  # Chuyển mô hình sang chế độ đánh giá (evaluation mode)
    with torch.no_grad(): # Tắt tính toán gradient
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Độ chính xác của mô hình trên 10000 ảnh test: {accuracy:.2f}%')
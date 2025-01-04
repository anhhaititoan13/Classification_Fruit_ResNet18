import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image, ImageFile
import os
import matplotlib.pyplot as plt

# Đảm bảo các ảnh bị cắt ngắn cũng được tải
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Custom loader để bắt các ngoại lệ khi tải ảnh
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except Exception as e:
            print(f"Error loading image {self.imgs[index][0]}: {e}")
            return None

# Thiết lập các biến
data_dir = 'datasets'
batch_size = 32
num_classes = 5
num_epochs = 5
learning_rate = 0.001

# Chuẩn bị dữ liệu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Tạo dataset huấn luyện với custom loader
train_dataset = SafeImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)

# In thông báo khi bắt đầu nạp ảnh
print("[INFO] Đang nạp ảnh...")

# Loại bỏ các ảnh bị lỗi (None)
train_dataset.samples = [sample for sample in train_dataset.samples if sample is not None]

# Hàm ghi nhật ký trạng thái xử lý dữ liệu
def log_processing_status(dataset, interval=500):
    total_images = len(dataset)
    for i, _ in enumerate(dataset, start=1):
        if i % interval == 0:
            print(f"[INFO] Đã xử lý {i}/{total_images}")

# Ghi nhật ký trạng thái xử lý dataset
log_processing_status(train_dataset)

# Tạo DataLoader từ dataset huấn luyện
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Tải mô hình ResNet đã được huấn luyện trước và tùy chỉnh nó
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Định nghĩa hàm mất mát và tối ưu hóa
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Đặt mô hình vào thiết bị (GPU nếu có, nếu không thì CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Khởi tạo các danh sách để lưu thông số
train_losses = []
train_accuracies = []

# Vòng lặp huấn luyện
for epoch in range(num_epochs):
    model.train()  # Đặt mô hình ở chế độ huấn luyện
    running_loss = 0.0
    correct = 0
    total = 0

    # Vòng lặp qua từng batch dữ liệu
    for inputs, labels in train_loader:
        if inputs is None or labels is None:
            continue
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Đặt lại gradient của các tham số tối ưu hóa

        outputs = model(inputs)  # Truyền dữ liệu đầu vào qua mô hình
        loss = criterion(outputs, labels)  # Tính toán mất mát
        loss.backward()  # Lan truyền ngược để tính toán gradient
        optimizer.step()  # Cập nhật các tham số của mô hình

        running_loss += loss.item() * inputs.size(0)  # Cộng dồn mất mát
        _, predicted = torch.max(outputs, 1)  # Lấy dự đoán của mô hình
        total += labels.size(0)  # Cộng dồn tổng số mẫu
        correct += (predicted == labels).sum().item()  # Cộng dồn số mẫu đúng

    # Tính mất mát trung bình và độ chính xác cho epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)  # Lưu mất mát của epoch
    train_accuracies.append(epoch_acc)  # Lưu độ chính xác của epoch

    # In thông tin về mất mát và độ chính xác của epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

# Đánh giá mô hình và tạo classification report
model.eval()  # Đặt mô hình ở chế độ đánh giá
true_labels = []
predicted_labels = []
with torch.no_grad():  # Tắt gradient để tiết kiệm bộ nhớ và tăng tốc độ
    for inputs, labels in train_loader:
        if inputs is None or labels is None:
            continue
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)  # Truyền dữ liệu qua mô hình
        _, predicted = torch.max(outputs, 1)  # Lấy dự đoán của mô hình

        true_labels.extend(labels.cpu().numpy())  # Lưu nhãn thật
        predicted_labels.extend(predicted.cpu().numpy())  # Lưu nhãn dự đoán

# In báo cáo phân loại
print("[INFO]: Đánh giá mô hình....")
print(classification_report(true_labels, predicted_labels, target_names=train_dataset.classes))

# Lưu mô hình
torch.save(model.state_dict(), 'fruit_classifier_resnet2.pth')

# Vẽ biểu đồ
epochs_range = range(1, num_epochs + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_losses, label='Mất mát khi training')
plt.plot(epochs_range, train_accuracies, label='Độ chính xác khi training')
plt.xlabel('Epochs')
plt.ylabel('Giá trị')
plt.title('Các thông số trong quá trình huấn luyện fruits')
plt.legend()
plt.tight_layout()
plt.savefig('training_results.png')
plt.show()

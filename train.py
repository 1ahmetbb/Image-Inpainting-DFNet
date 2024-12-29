import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from DFNet_model import DFNet  # DFNet modelini import et
from dataset import ImageInpaintingDataset  # Dataset sınıfını içeren dosya (eğer ayrı yazılacaksa)

# VGG tabanlı Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16].eval()  # İlk 16 katman
        for param in vgg.parameters():
            param.requires_grad = False  # VGG ağırlıkları sabitlenir
        self.vgg = vgg

    def forward(self, input, target):
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return nn.functional.mse_loss(input_features, target_features)

# Eğitim parametreleri
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model {device} üzerinde çalışacak.")

num_epochs = 10
batch_size = 16
learning_rate = 0.0005
checkpoint_dir = "checkpoints"

# Model, loss ve optimizer tanımları
model = DFNet().to(device)
criterion = nn.MSELoss()  # MSE Loss
perceptual_loss = PerceptualLoss().to(device)  # Perceptual Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dataset ve DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])
original_dir = "data/test_resimleri"  # Orijinal resimler klasörü
masked_dir = "data/test_resimleri_maskeli"  # Maskelenmiş resimler klasörü
dataset = ImageInpaintingDataset(original_dir, masked_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Eğitim döngüsü
start_time = time.time()

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        masked_imgs, original_imgs = data
        masked_imgs, original_imgs = masked_imgs.to(device), original_imgs.to(device)

        # Modelin tahmini
        outputs = model(masked_imgs)

        # Loss hesaplama
        loss = 0.8 * criterion(outputs, original_imgs) + 0.2 * perceptual_loss(outputs, original_imgs)

        # Geri yayılım ve optimizasyon
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Her 10 adımda bir loss değerini yazdır
        if (i + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], "
                  f"Loss: {running_loss / 10:.6f}, Geçen Süre: {elapsed_time // 60:.0f} dakika {elapsed_time % 60:.0f} saniye")
            running_loss = 0.0

    # Her epoch sonunda modeli kaydet
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"DFNet_epoch_{epoch + 1}.pth"))
    print(f"Epoch {epoch + 1} tamamlandı, model kaydedildi.")

# Eğitim tamamlandıktan sonra modeli kaydet
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "trained_dfnet.pth"))
print("Eğitim tamamlandı. Model başarıyla kaydedildi.")
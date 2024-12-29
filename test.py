import os
import cv2
import torch
import matplotlib.pyplot as plt
from DFNet_model import DFNet

def test_inpainting(test_image_path, model, device):
    """
    Maskelenmiş bir görüntü üzerinde model tahminini çalıştırır ve sonucu görselleştirir.
    """
    # Test resmi yükle
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Hata: Resim yüklenemedi - {test_image_path}")
        return
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB) / 255.0
    test_image = torch.tensor(test_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Model tahmini
    with torch.no_grad():
        output = model(test_image)

    # Çıkışı numpy formatına dönüştür
    output_img = output.squeeze().permute(1, 2, 0).cpu().numpy()

    # Görselleştir
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB))
    plt.title("Maskelenmiş Görüntü")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(output_img)
    plt.title("Tamamlanmış Görüntü")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Ana test süreci
if __name__ == "__main__":
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model {device} üzerinde çalışacak.")

    # Modeli yükle
    model = DFNet().to(device)
    model.load_state_dict(torch.load("checkpoints/trained_dfnet.pth", map_location=device))
    model.eval()

    # Test edilecek görüntünün yolu
    test_image_path = "data/test_resimleri_maskeli/masked_088925.jpg" 

    # Test işlemini başlat
    test_inpainting(test_image_path, model, device)
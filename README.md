# **IMAGE-INPAINTING-DFNET Projesi**

Bu proje, **derin öğrenme tabanlı görüntü inpainting** (eksik bölge doldurma) amacıyla geliştirilmiştir. DFNet modeli kullanılarak, **maskelenmiş (eksik bölge içeren) görüntülerin kayıp kısımlarını tahmin eder** ve tam bir görüntü oluşturur.

---

## **Proje İçeriği**

1. **DFNet Modeli:** Encoder-Decoder mimarisine dayalı bir model.
2. **Veri Ön İşleme:** Resimlere rastgele elips maskeler uygulanır.
3. **Model Eğitimi:** Maskelenmiş resimler kullanılarak model eğitilir.
4. **Model Testi:** Eğitilmiş model ile maskelenmiş resimlerin maskesiz halleri tahmin edilir.

---

## **Kullanılan Teknolojiler**

- **Python 3.8+**
- **PyTorch** (Derin Öğrenme Kütüphanesi)
- **OpenCV** (Görüntü İşleme)
- **Matplotlib** (Sonuçları Görselleştirme)

---

## **Kurulum Adımları**

1. **Projeyi Klonlayın:**
	```bash
 	git clone https://github.com/1ahmetbb/IMAGE-INPAINTING-DFNET.git
 	cd DFNet-ImageInpainting

2. **Gerekli Kütüphaneleri Kurun:**
	```bash
 	pip install -r requirements.txt

3. **Veri Setini Hazırlayın:**
	•	Orijinal resimlerinizi bir klasöre koyun. Örneğin: data/original_images.
	•	Maskelenmiş resimleri oluşturmak için mask_and_save_images fonksiyonunu kullanın.

4. **Modeli Eğitin:**
	```bash
 	python train.py
    
    •	Eğitilmiş model ağırlıkları trained_dfnet.pth dosyasına kaydedilecektir.

5. **Modeli Test Edin:**
	```bash
 	python test.py

## PROJE KULLANIMI

1. **Egitim:**
	•	Orijinal ve maskelenmiş resimlerden oluşan bir veri kümesiyle modeli eğitin.
2. **Test:**
	•	Maskelenmiş bir görüntü ile modelin dolgu yeteneğini test edin.
## Gereksinimler

Projede kullanilan temek kutuphaneler:

	•torch => 1.12.0
 	•torchvision => 0.13.0
  	•opencv-python
   	•matplptlib
    	•numpy
     
Bu kütüphaneleri kurmak için:

	```bash
 	pip install -r requirements.txt


## Egitilmis Model Agirliklari
Eğitilmiş model ağırlıklarını checkpoints/trained_dfnet.pth dosyasından yükleyebilirsiniz. Bu dosya, eğitilmiş DFNet modelini içerir ve yeniden eğitim gerektirmez.

## Gelecek Gelistirmeler
	•	Daha karmaşık maskelerle (ör. serbest çizim maskeleri) model performansını değerlendirme.
	•	Farklı veri setleriyle eğiterek daha geniş bir uygulama yelpazesi sağlama.
	•	Daha karmaşık model mimarilerini (ör. dikkat mekanizmaları) entegre etme.

## Katkida Bulunma 
**Katkida bulunmak icin:**
1.	Bu depoyu fork edin.
2.	Yeni bir özellik ekleyin veya bir hatayı düzeltin.
3.	PR (Pull Request) gönderin.

## Lisans
Bu proje MIT lisansı altında yayınlanmıştır. Ayrıntılar için LICENSE dosyasına bakın.

# Katarakt Tespiti Projesi

Bu proje, makine öğrenimi ve derin öğrenme tekniklerinin kullanıldığı bir katarakt tespit sistemini içermektedir. Proje kapsamında, kataraktlı ve normal gözlere ait görüntüler kullanılarak beş farklı görüntü tabanlı transformatör modeli (Google ViT, Microsoft BeiT, DEViT, LeViT ve Swin) geliştirilmiş ve performansları değerlendirilmiştir.

## Kurulum

Projenin çalıştırılması için Python ve bazı kütüphanelerin kurulu olması gerekmektedir. Aşağıdaki adımları izleyerek projeyi kurabilirsiniz:

1. Depoyu klonlayın:

git clone [https://github.com/kullaniciadi/katarakt-tespiti.git](https://github.com/musttoprak/Machine-Learning-Cataract-Detection.git)

2. Gerekli kütüphaneleri yükleyin:
  pip install torch torchvision
  pip install scikit-learn
  pip install timm
  pip install matplotlib

3. Kullanım
Projenin kullanımı için aşağıdaki adımları izleyebilirsiniz:

a.Veri setini indirin ve projenin klasörüne yerleştirin.
b.Veri setini işlemek ve modeli eğitmek için ilgili modeli models klasörü içerisinden seçip dosyasını çalıştırın.
c.Sonuçları görselleştirmek için results/result_with_graphics.py dosyasını çalıştırın.
d.Sonuçların ortalamasını görmek için results/result_with_text.py dosyasını çalıştırın.

4.Rapor
Projenin detaylı raporuna [buradan](MachineLearning-CataractDetection.pdf) ulaşabilirsiniz.

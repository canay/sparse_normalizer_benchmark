# Öğrenme oranı kontrolü: tamamlanan sonuç

210/210 koşu ve 3.060 epoch tamamlandı; 210 denemede yeniden koşu veya bilimsel hata kaydı yok. Önceden biten 92 koşu korundu, kalan 118 koşu tamamlandı. Eğitim 30 Eylül 2026 saat 23:55:20'de, bağımsız kontrol 1 Ekim saat 00:24:18'de (Türkiye saati) bitti.

Bağımsız Claude Opus 5.5 incelemesi, 420 metrik kaydını ve 28 eşleştirilmiş karşılaştırmayı yeniden hesaplayarak doğruladı. Karar: sonuç, aşağıdaki sınırlamalarla makaleye eklenebilir. Bu kayıt bir makale revizyonu veya tüm makalenin bilimsel kapanışı değildir.

Kontrol, sabit öğrenme oranını her güncellemede bütçeye göre doğrusal azalan bir oranla değiştirdi. Başlangıç oranı 0,001; warmup veya ek ayar araması yok. Aynı kompakt CIFAR-10 modeli, eğitim havuzundan ayrılan doğrulama verisi ve 10 eşleştirilmiş seed kullanıldı. Resmî test verisine erişilmedi.

| Top-k oranı | 3 epoch: dense'a göre fark | Nominal %95 aralık | 30 epoch: dense'a göre fark | Nominal %95 aralık |
|---|---:|---:|---:|---:|
| 0,125 | +4,332 yüzde puan | +3,796 ile +4,868 | −2,081 yüzde puan | −2,682 ile −1,480 |
| 0,25 | +3,127 yüzde puan | +2,347 ile +3,907 | −0,930 yüzde puan | −1,620 ile −0,240 |

Bu tek alternatif ayarda iki oran için de erken avantaj/geç dezavantaj işaret örüntüsü sürdü. Aralıklar 10 eşleştirilmiş seed için nominal, çoklu karşılaştırma düzeltmesi yapılmamış t aralıklarıdır. Ara bütçeler de raporlandı; tüm 28 karşılaştırma `ALL_BUDGET_PAIRED_CONTRASTS.csv` içindedir. İşaret örüntüsü her seed için aynı sonuç iddiası değildir.

Azalan oran, 21 yöntem×bütçe kombinasyonunun tamamında sabit referansa göre mutlak doğruluğu düşürdü. Dense'ın 30-epoch ortalaması %65,429; kayıtlı %60 kontrol tabanının üzerindedir. Bu yüzden sonuç genel öğrenme oranı dayanıklılığı, daha iyi bir optimizasyon politikası veya nedensel mekanizma kanıtı olarak sunulamaz. AdamW'nin ayrık weight-decay etkisi de öğrenme oranıyla birlikte değişir. İki politika arasındaki kaymalar betimseldir; bu farklar için ayrı bir önkayıtlı etki tahmini yoktur.

Sabit oran referansı eski 30 koşunun aynı seed yollarındaki önekleridir; yeni bağımsız replikasyon değildir. 20-seed havuzu oluşturulmadı. İlk 92 koşudan sonraki 118 koşu, kullanıcının yalnız boş VRAM/disk alt sınırlarını kaldıran yetkisiyle paylaşılan GPU'da tamamlandı: E12'de 28/30, E18/E24/E30'un tamamı. Kaynak, config ve runtime kimliği değişmedi; kaynak paylaşımının sayısal eşdeğerliği ayrıca test edilmedi. Bu iki çalışma rejiminin süreleri performans üstünlüğü karşılaştırması değildir.

Ham kayıtlardaki uygulanamaz `alpha_mean`/`alpha_std` alanları `NaN` içerir; bilimsel olarak kullanılan alanlar sonludur. Bunun protokolün bütün satırı sonlu tarif eden cümlesiyle sözcüğü sözcüğüne uyumsuzluğu korunur. Ayrıntı `../reviews/RAW_RECORD_SCHEMA_NOTES.md` dosyasındadır; orijinal kayıtlar değiştirilmedi.

M1/M2/M3/N3'ün kayıtlı başarısız kararları değişmedi. Bu kontrol yeni bir kurtarma deneyi değildir. Yeni deney başlatmaya gerek yoktur; sonraki iş, mevcut yetki ve canlı yazım rotası altında bu sınırlı sonucu makaleye işlemektir. Şu anki 20+9 sayfalık PDF'ler değişmedi ve bu yeni kontrol henüz PDF'ye eklenmedi. Tüm makalenin bilimsel kapanışı, Figure 1'in açık bulguları, dergi katkı riski ve son PDF yazar onayı ayrı ve açık kalır.

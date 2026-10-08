# N3-A main/forks: kalıcı arşiv konumu ve geri açma

1 Ekim 2026: açık yazar talimatıyla tamamlanmış N3-A deneyinin 252.928 dosyası kayıpsız ZIP64 arşivlendi. Bütün dosyaların CRC ve SHA-256 değerleri özgün yerel kayıtlarla doğrulandı; VPS kayıtlarının tamamı aynı dosya hashleriyle ayrıca eşleştirildi. Doğrulamadan sonra açık `outputs/main` ve `outputs/forks` ağaçları lokalde ve VPS’de kaldırıldı. Deneyi yeniden başlatmayın.

Kalıcı arşiv dizini: `experiments/2026-09-09_n3a_v2_vps_cpu/archived_outputs/20261001_n3a_lossless/` (makale köküne göre).

| Arşiv | Dosya | Özgün byte | ZIP byte | SHA-256 |
|---|---:|---:|---:|---|
| `main.zip` | 32768 | 4945071915 | 4459087759 | `a697052da2fc21bc5c96f14a211d4d67cf006ef3023d7eb5cd39d3305ba687bb` |
| `forks.zip` | 220160 | 8210593082 | 7258264072 | `31b4ce9756fefa7e374e22e00f667dd49eadafe4253b54966b8cfe54fd624c65` |

İki ZIP toplam 11,717,351,831 byte; hash listeleri, makbuzlar ve saklanan özgün VPS izin/zaman kayıtları dahil 11,843,469,509 byte. Özgün dosyalar 13,155,664,997 byte idi; bu kapsamda net mantıksal tasarruf 1,312,195,488 byte (GB = 10^9 byte). Yeni audit betikleri/state günlükleri bu küçük fark hesabına dahil değildir. Fiziksel disk alanı ve tüm cihazların boş alanı için bu mantıksal farkı ölçülmüş tahsis farkı gibi kullanmayın.

`*.files.jsonl` dosya yolu, boyut, SHA-256 ve yerel zaman/izin kaydını içerir; ZIP içinde birebir `_archive_manifest.jsonl` bulunur. `*.verified.json` tüm dosyaların doğrulama makbuzudur; `*.unpacked_removed.json` açık yerel kopyaların kaldırılmasını kaydeder. `forks.published.json` doğrulanmış geçici arşivin kalıcı Syncthing kapsamına alınmasını kaydeder; kamusal yayın anlamına gelmez.

Özgün Linux izin, uid/gid, inode, tahsis ve nanosecond zaman kayıtları makale kökündeki `q1-audit/snd-n3a-lossless-storage-archive-20261001/main.VPS_original_metadata.jsonl` ve `forks.VPS_original_metadata.jsonl` dosyalarındadır. Bunlar ZIP dosya içeriğini değiştirmez. Geri açma betiği dosya byte’larını ve yerel manifest zamanlarını geri koyar; özgün Linux sahiplik/izinlerini otomatik değiştirmez.

## Geri açma (gerektiğinde, eğitim başlatmadan)

Betik mevcut hedefin üstüne yazmaz; arşiv/hash listesini ve çıkarılan her dosyanın hashini doğrular. Arşivler korunur. Windows PowerShell:

```powershell
python "<controlled-workspace>/CALISMALAR/60_GONDERILDI/SCI-sparse_normalizer_benchmark/experiments/2026-09-09_n3a_v2_vps_cpu/restore_archived_outputs.py" --population main
python "<controlled-workspace>/CALISMALAR/60_GONDERILDI/SCI-sparse_normalizer_benchmark/experiments/2026-09-09_n3a_v2_vps_cpu/restore_archived_outputs.py" --population forks
```

VPS (önce her iki ZIP’in eşitlemesinin bitmesini ve kayıtlı hashleri doğrulayın):

```bash
python3 "<vps-home>/DOCS/AKADEMIK/CALISMALAR/60_GONDERILDI/SCI-sparse_normalizer_benchmark/experiments/2026-09-09_n3a_v2_vps_cpu/restore_archived_outputs.py" --population main
python3 "<vps-home>/DOCS/AKADEMIK/CALISMALAR/60_GONDERILDI/SCI-sparse_normalizer_benchmark/experiments/2026-09-09_n3a_v2_vps_cpu/restore_archived_outputs.py" --population forks
```

İstenen nüfus özgün `outputs/main` veya `outputs/forks` yoluna açılır. Bu yolların şimdi bulunmaması deney kayıtlarının kaybolduğu anlamına gelmez. Analiz için tek dosya gerektiğinde doğrulanmış ZIP’den o üye okunabilir; eğitim/refit/yeniden test yapılmaz.

## Eşitleme ve bilimsel sınır

Son VPS okumasında iki açık ağaç yoktur ve bu iki ağacın `.stversions` kopyaları 0’dır. `main.zip` VPS’de görünmektedir; her iki arşivin VPS/diğer bilgisayarlara tam varışı ve son hashleri henüz tasdik edilmemiştir. Arşivler Syncthing kapsamındadır; kaynakların birebir silinmesi VPS’de ayrıca doğrulanarak yapıldı, diğer projelerin sürüm geçmişi ve genel eşitleme ayarları değiştirilmedi. Diğer cihazların sürümleme ayarları fiziksel tasarrufu etkileyebilir.

`SCIENTIFIC_DECISION.json`, analiz özetleri, giriş verileri ve kayıtlı başarısız bilimsel karar canlı yerlerinde korunur. N3-A 6144/6144 tamamlanmış `KILLED_FOR_REGISTERED_EPSILON`; N3-B kayıtlı durdurma nedeniyle başlatılmamıştır. Arşivleme makale bilimsel kapanışı, kişisel PDF onayı, gönderim paketi veya kamusal veri yayını değildir.

Tam işlem ve geri açma smoke kanıtı: `q1-audit/snd-n3a-lossless-storage-archive-20261001/ACTUAL_STORAGE_COMPLETION_RECEIPT.json`. İlk boş dizin sıralama hatası ve ara eşitleme/SCP hataları audit içinde korunur; başarılı son makbuzlar önceki hataları PASS yapmaz.

## Sonraki VPS varış doğrulaması

2026-10-01T15:27:55Z tarihli ayrı salt-okunur ölçümde `main.zip` tam byte/SHA-256 eşleşmesiyle doğrulandı. `forks.zip` henüz VPS’de bulunmuyordu; iki arşivin ve diğer cihazların eşitlemesi tamamlandı sayılmıyor. İki açık ağaç hâlâ yok ve özgün bilimsel karar hashi değişmedi. Bu sonraki ölçüm, yukarıdaki tarihsel kontrolü ve ilk tamamlanma makbuzunu geriye dönük değiştirmez. Kanıt: `q1-audit/snd-n3a-lossless-storage-archive-20261001/ACTUAL_VPS_ARCHIVE_ARRIVAL_SNAPSHOT.json` ve `VPS_ARCHIVE_ARRIVAL_TRANSPORT.json`.

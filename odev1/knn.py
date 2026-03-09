import numpy as np
import pickle
import os

# =============================================================================
# CIFAR-10 SINIF İSİMLERİ (10 sınıf)
# =============================================================================
SINIF_ISIMLERI = [
    "uçak",      # 0
    "otomobil",  # 1
    "kuş",       # 2
    "kedi",      # 3
    "geyik",     # 4
    "köpek",     # 5
    "kurbağa",   # 6
    "at",        # 7
    "gemi",      # 8
    "kamyon"     # 9
]

# =============================================================================
# VERİ YOLU — knn.py'nin yanındaki cifar-10-batches-py/ klasörü
# Yoksa kullanıcıdan klasör yolu istenir.
# =============================================================================
varsayilan_yol = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar-10-batches-py")

if os.path.isdir(varsayilan_yol):
    cifar_klasoru = varsayilan_yol
    print(f"✅ CIFAR-10 klasörü bulundu: {cifar_klasoru}")
else:
    print("⚠️  cifar-10-batches-py klasörü bulunamadı.")
    print("   CIFAR-10 verisini buradan indirin: https://www.cs.toronto.edu/~kriz/cifar.html")
    cifar_klasoru = input("CIFAR-10 klasörünün tam yolunu girin: ").strip()

egitim_dosyasi = os.path.join(cifar_klasoru, "data_batch_1")
test_dosyasi   = os.path.join(cifar_klasoru, "test_batch")

if not os.path.isfile(egitim_dosyasi):
    print(f"❌ Eğitim dosyası bulunamadı: {egitim_dosyasi}")
    input("Çıkmak için Enter'a basın...")
    raise SystemExit

if not os.path.isfile(test_dosyasi):
    print(f"❌ Test dosyası bulunamadı: {test_dosyasi}")
    input("Çıkmak için Enter'a basın...")
    raise SystemExit

# =============================================================================
# VERİ OKUMA (CIFAR-10 formatı: b'data', b'labels')
# =============================================================================
print("\n📦 Eğitim verisi yükleniyor...")
with open(egitim_dosyasi, 'rb') as f:
    egitim_dict = pickle.load(f, encoding='bytes')

print("📦 Test verisi yükleniyor...")
with open(test_dosyasi, 'rb') as f:
    test_dict = pickle.load(f, encoding='bytes')

egitim_verisi_ham     = egitim_dict[b'data']    # (10000, 3072) — her satır bir görüntü
egitim_etiketleri_ham = egitim_dict[b'labels']  # Liste, 0-9 arası etiketler

test_verisi_ham       = test_dict[b'data']      # (10000, 3072)
test_etiketleri_ham   = test_dict[b'labels']    # Liste, 0-9 arası etiketler

# =============================================================================
# KULLANICI ETKİLEŞİMİ — tüm parametreler input() ile alınır
# =============================================================================
print("\n" + "="*60)
print("        CIFAR-10  k-NN  SINIFLANDIRICI")
print("="*60)

# Kaç eğitim örneği kullanılacak?
EGITIM_SINIRI = 0
while EGITIM_SINIRI < 1 or EGITIM_SINIRI > 10000:
    try:
        EGITIM_SINIRI = int(input("\nKaç eğitim örneği kullanılsın? (1 - 10000, önerilen: 1000): ").strip())
        if EGITIM_SINIRI < 1 or EGITIM_SINIRI > 10000:
            print("1 ile 10000 arasında bir değer girin.")
    except ValueError:
        print("Geçersiz giriş, tam sayı girin.")

# Kaç test örneği denenecek?
TEST_SINIRI = 0
while TEST_SINIRI < 1 or TEST_SINIRI > 10000:
    try:
        TEST_SINIRI = int(input("Kaç test örneği denensin? (1 - 10000, önerilen: 10): ").strip())
        if TEST_SINIRI < 1 or TEST_SINIRI > 10000:
            print("1 ile 10000 arasında bir değer girin.")
    except ValueError:
        print("Geçersiz giriş, tam sayı girin.")

# Mesafe metriği
metrik = ""
while metrik not in ["L1", "L2"]:
    metrik = input("Mesafe metriği seçin (L1 veya L2): ").strip().upper()
    if metrik not in ["L1", "L2"]:
        print("Geçersiz seçim! 'L1' veya 'L2' girin.")

# k değeri
k = 0
while k < 1:
    try:
        k = int(input("k değerini girin (örn: 3): ").strip())
        if k < 1:
            print("k en az 1 olmalıdır!")
    except ValueError:
        print("Geçersiz giriş, tam sayı girin.")

# =============================================================================
# VERİ HAZIRLAMA
# =============================================================================
egitim_verisi     = np.array(egitim_verisi_ham[:EGITIM_SINIRI],     dtype=np.float64)
egitim_etiketleri = np.array(egitim_etiketleri_ham[:EGITIM_SINIRI], dtype=np.int32)

test_verisi     = np.array(test_verisi_ham[:TEST_SINIRI],     dtype=np.float64)
test_etiketleri = np.array(test_etiketleri_ham[:TEST_SINIRI], dtype=np.int32)

print(f"\n✅ Eğitim verisi : {egitim_verisi.shape[0]} örnek")
print(f"✅ Test verisi   : {test_verisi.shape[0]} örnek")
print(f"\nMetrik : {metrik}   |   k : {k}")
print(f"\n{'='*65}")
print("  k-NN TAHMİNLERİ BAŞLIYOR...")
print(f"{'='*65}\n")

# =============================================================================
# k-NN ALGORİTMASI — fonksiyon kullanılmadan, düz döngü
# =============================================================================
dogru_sayisi = 0

for i in range(TEST_SINIRI):
    test_ornegi   = test_verisi[i]       # (3072,) piksel vektörü
    gercek_etiket = int(test_etiketleri[i])

    # ---- Mesafe hesaplama (numpy broadcast — iç döngü yok) ----
    if metrik == "L1":
        # Manhattan: |x1 - x2| toplamı
        mesafeler = np.sum(np.abs(egitim_verisi - test_ornegi), axis=1)
    else:
        # Öklid: sqrt( Σ(x1-x2)² )
        mesafeler = np.sqrt(np.sum((egitim_verisi - test_ornegi) ** 2, axis=1))

    # ---- En yakın k komşuyu bul ----
    en_yakin_k_indis        = np.argsort(mesafeler)[:k]
    k_komsularin_etiketleri = egitim_etiketleri[en_yakin_k_indis]

    # ---- Çoğunluk oylaması ----
    etiket_sayilari = np.bincount(k_komsularin_etiketleri, minlength=10)
    tahmin_etiket   = int(np.argmax(etiket_sayilari))

    # ---- Sonuç ----
    if tahmin_etiket == gercek_etiket:
        dogru_sayisi += 1
        durum = "✓ DOĞRU"
    else:
        durum = "✗ YANLIŞ"

    print(f"  Test #{i+1:>4d}  |  "
          f"Tahmin: {tahmin_etiket} - {SINIF_ISIMLERI[tahmin_etiket]:<10}  |  "
          f"Gerçek: {gercek_etiket} - {SINIF_ISIMLERI[gercek_etiket]:<10}  |  "
          f"{durum}")

# =============================================================================
# SONUÇ
# =============================================================================
basari_orani = (dogru_sayisi / TEST_SINIRI) * 100
print(f"\n{'='*65}")
print(f"  SONUÇ:")
print(f"  Doğru tahmin    : {dogru_sayisi} / {TEST_SINIRI}")
print(f"  Başarı oranı    : %{basari_orani:.1f}")
print(f"  Kullanılan metrik: {metrik}")
print(f"  k değeri        : {k}")
print(f"  Eğitim seti     : {EGITIM_SINIRI} örnek")
print(f"{'='*65}")

# 📈 Borsa Tahmin ve Portföy Yönetim Sistemi

Yapay zeka destekli borsa tahmin ve portföy yönetim uygulaması. BIST hisseleri için fiyat tahminleri yapın, teknik analizleri inceleyin ve portföyünüzü profesyonelce yönetin.

## 🚀 Özellikler

- **Akıllı Tahminler**: Makine öğrenimi ile gelecek fiyat tahminleri
- **Teknik Analiz**: MA20, MA50, RSI göstergeleri ile detaylı analiz
- **Portföy Takibi**: Hisselerinizi ekleyin, performansınızı izleyin
- **Görsel Grafikler**: İnteraktif fiyat ve gösterge grafikleri
- **Kullanıcı Yönetimi**: Güvenli kayıt ve giriş sistemi

## 🛠️ Teknolojiler

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Veritabanı**: SQLite
- **Veri Kaynağı**: Yahoo Finance API
- **ML**: scikit-learn (RandomForestRegressor)

## ⚙️ Kurulum

### Gereksinimler

- Python 3.9+
- pip (Python paket yöneticisi)

### Adımlar

1. Projeyi klonlayın:

```bash
git clone https://github.com/kullaniciadi/borsa-tahmin-sistemi.git
cd borsa-tahmin-sistemi
```

2. Sanal ortam oluşturun:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac için
venv\Scripts\activate     # Windows için
```

3. Gereksinimleri yükleyin:

```bash
pip install -r requirements.txt
```

4. Uygulamayı başlatın:

```bash
python app.py
```

5. Tarayıcınızda açın:

```
http://127.0.0.1:5001
```

## 📱 Kullanım

1. **Kayıt/Giriş**

   - Yeni hesap oluşturun veya mevcut hesabınızla giriş yapın

2. **Hisse Tahmini**

   - Hisse kodunu girin (örn: THYAO.IS)
   - Tahmin tarihini seçin
   - "Tahmin Et" butonuna tıklayın

3. **Portföy Yönetimi**
   - "Portföy" sekmesinden hisse ekleyin
   - Alış bilgilerini girin
   - Performans takibini yapın

## 💾 Veritabanı Yapısı

### Users Tablosu

| Alan     | Tür     | Açıklama      |
| -------- | ------- | ------------- |
| id       | INTEGER | Primary Key   |
| username | TEXT    | Kullanıcı Adı |
| password | TEXT    | Şifre         |

### Portfolios Tablosu

| Alan           | Tür     | Açıklama    |
| -------------- | ------- | ----------- |
| id             | INTEGER | Primary Key |
| user_id        | INTEGER | Foreign Key |
| symbol         | TEXT    | Hisse Kodu  |
| quantity       | INTEGER | Adet        |
| purchase_price | REAL    | Alış Fiyatı |
| purchase_date  | DATE    | Alış Tarihi |

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin yeni-ozellik`)
5. Pull Request oluşturun

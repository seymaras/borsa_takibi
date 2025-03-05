from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Hugging Face veri setini yükle
dataset = load_dataset("habanoz/eco-news-tr")

# Veriyi pandas DataFrame'e çevir
df = dataset['train'].to_pandas()

# "Borsa İstanbul" veya "BIST" içeren haberleri filtrele
borsa_haberleri = df[df["text"].str.contains("Borsa İstanbul|BIST", case=False, na=False)]

# İlk 5 haberi gösterelim
print("\n📊 **Borsa İstanbul Haberleri:**")
print(borsa_haberleri.head())

# Model ve tokenizer'ı yükle
model_name = "savasy/bert-base-turkish-sentiment-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Cihazı belirle
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# Pipeline'ı oluştur
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Duygu analizi fonksiyonu
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])[0]  # Metni 512 karakterle sınırla
        return result['label']
    except Exception as e:
        print(f"Hata: {text[:100]}... için analiz yapılamadı: {str(e)}")
        return "NEUTRAL"

class NewsAnalyzer:
    def __init__(self):
        self.model_name = "savasy/bert-base-turkish-sentiment-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.sentiment_pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, framework="pt")

        # Haber veri setini yükle
        self.dataset = load_dataset("habanoz/eco-news-tr")
        self.df = self.dataset['train'].to_pandas()

    def get_news_sentiment(self, hisse_adi):
        """
        Kullanıcının belirttiği hisse senedi ile ilgili haberleri çekip duygu analizi yapar.
        """
        # Hisse adı içeren haberleri filtrele
        hisse_haberleri = self.df[self.df["text"].str.contains(hisse_adi, case=False, na=False)]
        
        # Eğer haber yoksa
        if hisse_haberleri.empty:
            return {"status": "error", "message": f"{hisse_adi} için haber bulunamadı.", "sentiments": [], "sentiment_score": 0}

        # Duygu analizi uygula
        hisse_haberleri["sentiment"] = hisse_haberleri["text"].apply(lambda x: self.sentiment_pipeline(x)[0]['label'])

        # Pozitif, negatif ve nötr haber sayılarını hesapla
        sentiment_counts = hisse_haberleri["sentiment"].value_counts().to_dict()

        # Sentiment skorunu hesapla (Pozitif: +1, Nötr: 0, Negatif: -1)
        sentiment_score = (
            sentiment_counts.get("positive", 0) - sentiment_counts.get("negative", 0)
        ) / (sum(sentiment_counts.values()) + 1e-6)

        return {
            "status": "success",
            "hisse": hisse_adi,
            "sentiments": sentiment_counts,
            "sentiment_score": sentiment_score,
            "news": hisse_haberleri[["text", "sentiment"]].to_dict(orient="records")
        }
# Haberlerin duygu analizini yap
borsa_haberleri["sentiment"] = borsa_haberleri["text"].apply(analyze_sentiment)

# Sonuçları göster
print("\n📊 **Borsa İstanbul Haber Analizi:**")
print(borsa_haberleri[["text", "sentiment"]].head())

# Sentiment dağılımını görselleştir
plt.figure(figsize=(10,6))
sentiment_counts = borsa_haberleri["sentiment"].value_counts()
sentiment_counts.plot(kind="bar", color=['green', 'red', 'gray'])
plt.title("Borsa İstanbul Haberlerinin Duygu Dağılımı")
plt.xlabel("Duygu")
plt.ylabel("Haber Sayısı")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

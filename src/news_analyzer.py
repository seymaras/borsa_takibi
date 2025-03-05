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

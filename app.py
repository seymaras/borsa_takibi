from flask import Flask, json, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from routes.auth import auth_bp
from database.connection import init_db, create_connection
import os
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from functools import lru_cache
from src.news_analyzer import NewsAnalyzer


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_secret_key")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Oturum süresini 7 güns yap
app.register_blueprint(auth_bp)

# Veritabanna baglnma
def create_user_table():
    conn = create_connection("users.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()


def create_stocks_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        value DECIMAL(10, 2),
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)
    conn.commit()
    conn.close()

def create_portfolio_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        symbol TEXT NOT NULL,
        quantity INTEGER,
        purchase_price DECIMAL(10, 2),
        purchase_date DATE,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)
    conn.commit()
    conn.close()

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Global NewsAnalyzer instance
news_analyzer = NewsAnalyzer()

# Hisse tahmini için sayfa
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            data = request.get_json()
            symbol = data['symbol'].upper().strip()
            if not symbol.endswith('.IS'):
                symbol = f"{symbol}.IS"
            target_date = data['date']
            
            # LSTM ve RF tahminleri
            lstm_prediction, lstm_accuracy = predict_price_lstm(symbol, target_date)
            rf_prediction, rf_accuracy = predict_rf(symbol, target_date)
            
            # Haber analizi
            news_analysis = news_analyzer.get_news_sentiment(symbol)
            
            # Son fiyat ve teknik göstergeler
            df = yf.download(symbol, start=datetime.now() - timedelta(days=30), 
                           end=datetime.now(), progress=False)
            
            if df.empty:
                raise Exception("Hisse verisi bulunamadı")
                
            last_price = float(df['Close'].iloc[-1])
            
            # NaN kontrolü ile teknik göstergeler
            ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
            ma50 = df['Close'].rolling(window=50).mean().iloc[-1]
            rsi = calculate_rsi(df['Close']).iloc[-1]
            
            # NaN değerleri kontrol et ve varsayılan değerler ata
            ma20 = 0 if np.isnan(ma20) else float(ma20)
            ma50 = 0 if np.isnan(ma50) else float(ma50)
            rsi = 50 if np.isnan(rsi) else float(rsi)
            
            # Trend ve RSI sinyalleri
            trend = "Yükseliş" if ma20 > ma50 else "Düşüş"
            rsi_signal = "Aşırı Alım" if rsi > 70 else "Aşırı Satım" if rsi < 30 else "Nötr"
            
            # Tahminleri birleştir
            base_prediction = (lstm_prediction + rf_prediction) / 2
            
            # Haber etkisi
            sentiment_adjustment = 0
            if news_analysis['status'] == 'success':
                sentiment_adjustment = news_analysis['sentiment_score'] * 0.02
                final_prediction = base_prediction * (1 + sentiment_adjustment)
            else:
                final_prediction = base_prediction
            
            # Değişim oranı
            price_change = ((final_prediction - last_price) / last_price) * 100
            
            return jsonify({
                "symbol": symbol,
                "current_price": round(last_price, 2),
                "predicted_price": round(final_prediction, 2),
                "price_change": round(price_change, 2),
                "target_date": target_date,
                "technical_analysis": {
                    "ma20": round(ma20, 2),
                    "ma50": round(ma50, 2),
                    "rsi": round(rsi, 2),
                    "trend": trend,
                    "rsi_signal": rsi_signal
                },
                "accuracy_metrics": {
                    "lstm_accuracy": round(lstm_accuracy, 2),
                    "rf_accuracy": round(rf_accuracy, 2),
                    "combined_accuracy": round((lstm_accuracy + rf_accuracy) / 2, 2)
                },
                "news_analysis": news_analysis
            })
            
        except Exception as e:
            return jsonify({"error": f"Analiz yapılırken bir hata oluştu: {str(e)}"}), 400

    return render_template("tahmin.html")

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

# Kayıt ol
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = create_connection("users.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            
            # Kullanıcı ID'sini al
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            
            # Oturum bilgilerini ayarla
            session["username"] = username
            session["id"] = user[0]
            
            flash("Kayıt başarılı!", "success")
            return redirect(url_for("tahmin"))  # Tahmin sayfasına yönlendir
        except sqlite3.IntegrityError:
            flash("Bu kullanıcı adı zaten alınmış!", "danger")
        finally:
            conn.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = create_connection("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        if user:
            flash(f"Hoş geldin, {username}!", "success")
            session["username"] = username
            session["id"] = user[0]
            return redirect(url_for("tahmin"))
        else:
            flash("Kullanıcı adı veya şifre yanlış!", "danger")
        conn.close()

    return render_template("login.html")

#cıkış yapma
@app.route("/logout", methods=["POST"])
def logout():
    session.pop("username", None)
    flash("Başarıyla çıkış yaptınız.", "success")
    return redirect(url_for("tahmin"))

# Tahmin sayfası
@app.route("/tahmin")
def tahmin():
    if "username" not in session:
        flash("Lütfen önce giriş yapın!", "danger")
        return redirect(url_for("login"))
    return render_template('tahmin.html')
    return render_template("tahmin.html")

@app.route('/add_to_portfolio', methods=['POST'])
def add_to_portfolio():
    if "username" not in session:
        return jsonify({"error": "Lütfen giriş yapın"}), 401
    
    try:
        data = request.get_json()
        symbol = data.get('symbol').upper()
        if not symbol.endswith('.IS'):
            symbol = f"{symbol}.IS"
        quantity = data.get('quantity')
        purchase_price = data.get('purchase_price')
        
        conn = create_connection()
        cursor = conn.cursor()
        
        # Portföye hisse ekleme
        cursor.execute("""
        INSERT INTO portfolios (user_id, symbol, quantity, purchase_price, purchase_date)
        VALUES (?, ?, ?, ?, ?)
        """, (session['id'], symbol, quantity, purchase_price, datetime.now().date()))
        
        conn.commit()
        conn.close()
        
        return jsonify({"message": "Hisse portföye başarıyla eklendi"})
        
    except Exception as e:
        return jsonify({"error": f"Hisse eklenirken bir hata oluştu: {str(e)}"}), 400

@app.route('/get_portfolio', methods=['GET'])
def get_portfolio():
    if "username" not in session:
        return jsonify({"error": "Lütfen giriş yapın"}), 401
    
    try:
        conn = create_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT symbol, quantity, purchase_price, purchase_date 
        FROM portfolios 
        WHERE user_id = ?
        """, (session['id'],))
        
        portfolio = cursor.fetchall()
        
        portfolio_data = []
        for item in portfolio:
            symbol, quantity, purchase_price, purchase_date = item
            
            try:
                df = yf.download(symbol, start=datetime.now() - timedelta(days=1), 
                               end=datetime.now(), progress=False)
                current_price = float(df['Close'].iloc[-1])
                
                profit_loss = (current_price - purchase_price) * quantity
                profit_loss_percentage = ((current_price - purchase_price) / purchase_price) * 100
                
                portfolio_data.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "purchase_price": purchase_price,
                    "purchase_date": purchase_date,
                    "current_price": round(current_price, 2),
                    "profit_loss": round(profit_loss, 2),
                    "profit_loss_percentage": round(profit_loss_percentage, 2)
                })
            except:
                continue
        
        conn.close()
        return jsonify({"portfolio": portfolio_data})
    except Exception as e:
        return jsonify({"error": f"Portföy bilgileri alınırken bir hata oluştu: {str(e)}"}), 400

@app.route('/get_predictions')
def get_predictions():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, date, value 
            FROM stocks 
            WHERE user_id = ? 
            ORDER BY date DESC 
            LIMIT 5
        """, (session['id'],))
        predictions = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            "symbol": pred[0],
            "date": pred[1],
            "value": pred[2]
        } for pred in predictions])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def create_lstm_model(sequence_length):
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=(sequence_length, 5)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(units=64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber')
    return model

def prepare_data(df, sequence_length):
    features = ['Close', 'MA20', 'MA50', 'RSI', 'Volatility']
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['ATR'] = calculate_atr(df)
    
    dataset = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, 0])
    
    return np.array(X), np.array(y), scaler

def predict_price_lstm(symbol, target_date):
    try:
        # Veri çekme
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            raise Exception("Hisse senedi verisi bulunamadı")
        
        # Teknik göstergeleri hesapla
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # NaN değerleri doldur
        df = df.fillna(method='bfill')
        
        # Veriyi hazırla
        sequence_length = 60
        X, y, scaler = prepare_data(df, sequence_length)
        
        if len(X) < sequence_length:
            raise Exception("Yeterli veri yok")
        
        # Model oluştur ve eğit
        model = create_lstm_model(sequence_length)
        split = int(len(X) * 0.8)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
        
        model.fit(
            X[:split], 
            y[:split],
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Tahmin için son veriyi hazırla
        last_sequence = X[-1:]
        
        # Tahmin yap
        predicted_scaled = model.predict(last_sequence, verbose=0)
        predicted_price = scaler.inverse_transform(
            np.concatenate([predicted_scaled, np.zeros((1, 4))], axis=1)
        )[0, 0]
        
        # Doğruluk hesapla
        accuracy = calculate_accuracy(model, X[split:], y[split:])
        
        return predicted_price, accuracy
        
    except Exception as e:
        raise Exception(f"Tahmin yapılırken hata oluştu: {str(e)}")

def calculate_accuracy(model, X_test, y_test):
    try:
        predictions = model.predict(X_test, verbose=0)
        mse = np.mean((predictions - y_test) ** 2)
        accuracy = 100 * (1 - np.sqrt(mse))
        return max(0, min(100, accuracy))  # 0-100 arasında sınırla
    except:
        return 0

def predict_rf(symbol, target_date):
    # Veri çekme
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Özellikler
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df = df.fillna(method='bfill')
    
    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df[['MA20', 'MA50', 'RSI']].values[:-1]
    y = df['Close'].values[1:]
    
    # Eğitim ve tahmin
    model.fit(X, y)
    last_data = df[['MA20', 'MA50', 'RSI']].values[-1].reshape(1, -1)
    prediction = model.predict(last_data)[0]
    
    return prediction, calculate_accuracy(model, X[-len(y):], y)

@lru_cache(maxsize=32)
def get_stock_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date, progress=False)

@app.route('/get_live_data', methods=['POST'])
def get_live_data():
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        live_data = []
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                current_data = stock.history(period='1d')
                
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    previous_price = current_data['Open'].iloc[0]
                    price_change = ((current_price - previous_price) / previous_price) * 100
                    
                    live_data.append({
                        'symbol': symbol.replace('.IS', ''),
                        'price': round(current_price, 2),
                        'change': round(price_change, 2)
                    })
            except Exception as e:
                print(f"Hisse verisi alınamadı {symbol}: {str(e)}")
                continue
        
        return jsonify(live_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_borsa_data')
def get_borsa_data():
    try:
        symbols = [
            'THYAO.IS', 'GARAN.IS', 'ASELS.IS', 'KCHOL.IS', 'AKBNK.IS',
            'EREGL.IS', 'BIMAS.IS', 'SISE.IS', 'TUPRS.IS', 'YKBNK.IS'
        ]
        
        live_data = []
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                current_data = stock.history(period='1d', interval='1m')
                
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    open_price = current_data['Open'].iloc[0]
                    price_change = ((current_price - open_price) / open_price) * 100
                    volume = current_data['Volume'].sum()
                    
                    live_data.append({
                        'symbol': symbol.replace('.IS', ''),
                        'price': round(current_price, 2),
                        'change': round(price_change, 2),
                        'volume': int(volume),
                        'time': current_data.index[-1].strftime('%H:%M')
                    })
            except Exception as e:
                print(f"Hata: {symbol} için veri alınamadı - {str(e)}")
                continue
                
        return jsonify(live_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Program giriş noktası
if __name__ == "__main__":
    init_db()  # Veritabanını başlat
    create_user_table()
    create_stocks_table()
    create_portfolio_table()
    app.run(debug=True, port=5001)


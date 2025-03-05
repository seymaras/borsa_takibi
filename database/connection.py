import sqlite3
import os

def create_connection(db_name="users.db"):
    # instance klasörünü oluştur
    if not os.path.exists('instance'):
        os.makedirs('instance')
    
    # Veritabanı yolunu oluştur
    db_path = os.path.join('instance', db_name)
    return sqlite3.connect(db_path)

def init_db():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
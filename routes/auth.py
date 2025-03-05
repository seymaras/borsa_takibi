from flask import Blueprint, request, session, redirect, url_for, flash, render_template
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests
import os
from dotenv import load_dotenv
from database.connection import create_connection

auth_bp = Blueprint('auth', __name__)

load_dotenv()

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
client_secrets_file = "client_secrets.json"  # Google Cloud Console'dan indirdiğiniz dosya

flow = Flow.from_client_secrets_file(
    client_secrets_file,
    scopes=["openid", "email", "profile"],
    redirect_uri="http://127.0.0.1:5001/callback"
)

@auth_bp.route("/login/google")
def google_login():
    authorization_url, state = flow.authorization_url()
    session["state"] = state
    return redirect(authorization_url)

@auth_bp.route("/callback")
def callback():
    try:
        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials
        id_info = id_token.verify_oauth2_token(
            credentials.id_token, requests.Request(), GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10
        )
        
        username = id_info.get("email")
        if not username:
            flash("Google hesabından email bilgisi alınamadı!", "danger")
            return redirect(url_for("login"))
        
        conn = create_connection("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if not user:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                         (username, "google_auth"))
            conn.commit()
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
        
        session.clear()  # Mevcut oturum verilerini temizle
        session["username"] = username
        session["id"] = user[0]
        session.permanent = True  # Oturumu kalıcı yap
        
        flash(f"Google hesabınızla giriş yaptınız: {username}", "success")
        return redirect("/tahmin")  # Direkt URL kullanarak yönlendir
        
    except Exception as e:
        print(f"Google login error: {str(e)}")  # Hata ayıklama için
        flash("Google ile giriş yapılırken bir hata oluştu", "danger")
        return redirect(url_for("login")) 

import os
import sys
from pyngrok import ngrok
from streamlit.web import cli as stcli
STATIC_DOMAIN = None 

def main():
    print("Mempersiapkan public URL...")
    
    try:
        if STATIC_DOMAIN:
            public_url = ngrok.connect(8501, domain=STATIC_DOMAIN).public_url
        else:
            public_url = ngrok.connect(8501).public_url
            
        print(f"\n========================================================")
        print(f"  PUBLIC URL APP ANDA: {public_url}")
        print(f"========================================================\n")
    except Exception as e:
        print(f"Gagal membuat public URL: {e}")
        print("Pastikan Anda sudah login ke ngrok (ngrok config add-authtoken <token>) jika diperlukan.")
        return

    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()

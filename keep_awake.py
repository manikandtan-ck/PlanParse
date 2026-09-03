import os
import sys
from playwright.sync_api import sync_playwright

def ping_app():
    url = "https://planparse-wall-extraction.streamlit.app/" 
    
    print(f"Launching headless browser to ping: {url}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(url, wait_until="networkidle", timeout=60000)
            print("App loaded successfully. Title:", page.title())
        except Exception as e:
            print(f"Failed to ping app: {e}")
            sys.exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    ping_app()

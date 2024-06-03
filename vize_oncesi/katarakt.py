import os
import time
import requests
import base64
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Diller ve arama terimleri
languages = {
    'tr': 'katarakt göz fotoğrafları', # Türkçe
    'en': 'cataract eye images',   # İngilizce
    'fr': 'cataracte images de lœil',  # Fransızca
    'de': 'Katarakt Augenbilder',   # Almanca
    'es': 'catarata imágenes del ojo',   # İspanyolca
    'it': 'cataratta immagini dell occhio',  # İtalyanca
    'pt': 'catarata imagens do olho',   # Portekizce
    'ru': 'катаракта изображения глаза',  # Rusça
    'ar': 'ساد صور العين',        # Arapça
    'ja': '白内障の目の画像',       # Japonca
    'ko': '백내장 눈 이미지',       # Korece
    'zh-CN': '白内障眼睛图片',    # Çince (Basitleştirilmiş)
    'zh-TW': '白內障眼睛圖片',    # Çince (Geleneksel)
    'hi': 'मोतियाबिंद आंख की छवियाँ', # Hintçe
    'bn': 'অপুষ্টতা চোখ ছবি',    # Bengalce
    'ur': 'انتہائی کمی آنکھ تصاویر', # Urduca
    'ta': 'கண் தீவனம் படங்கள்', # Tamilce
    'te': 'నేత్రం పిండిక చిత్రాలు', # Teluguca
    'sq': 'Katarakte imazhe sy',   # Arnavutça
    'bg': 'Катаракт очи снимки',    # Bulgarca
    'hr': 'Katarakta slike oka',   # Hırvatça
    'da': 'Katarakt øje billeder',    # Danca
    'fi': 'Kataraktti silmä kuvat',  # Fince
    'el': 'Καταρράκτης φωτογραφίες ματιού',  # Yunanca
    'hu': 'Szürkehályog szem képek', # Macarca
    'no': 'Grå stær øye bilder'     # Norveççe
}


folder = 'cataract_images'  # Kaydedilecek klasör

# Chrome WebDriver'ı başlat
driver = webdriver.Chrome(ChromeDriverManager().install())

# Her dil için döngü yap
for lang_code, search_term in languages.items():
    print(f"Downloading images for {lang_code}...")
    
    # Google Görseller'e git
    driver.get('https://www.google.com/imghp?hl=' + lang_code)

    # Arama kutusunu bul ve arama yap
    search_box = driver.find_element(By.CLASS_NAME,'gLFyf')
    search_box.send_keys(search_term)
    search_box.send_keys(Keys.RETURN)

    # Sayfayı aşağı kaydır ve yüklenen sayfa
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Bekleme süresi
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # Tüm resimleri bul
    # Tüm resimleri bul
    div_container = driver.find_element(By.CLASS_NAME, 'islrc')
    images = div_container.find_elements(By.CSS_SELECTOR,'img')

    # Klasör oluştur (varsa atla)
    os.makedirs(folder, exist_ok=True)

    # Resimleri indir
    for i, img in enumerate(images[:100]):  # İlk 100 resmi al
        src = img.get_attribute('src')
        if src is not None and src.startswith('data:image/jpeg;base64'):
            # base64 kodunu al ve çöz
            base64_str = src.split(';base64,')[-1]
            img_data = base64.b64decode(base64_str)
            # Resmi kaydet
            with open(os.path.join(folder, f'{lang_code}_{i}.jpg'), 'wb') as f:
                f.write(img_data)
    
    print(f"{len(images[:100])} images downloaded for {lang_code}.")

# Tarayıcıyı kapat
driver.quit()

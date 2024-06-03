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
    'tr': 'normal göz fotoğrafları', # Türkçe
    'en': 'normal eye images',   # İngilizce
    'fr': 'images normales de lœil',  # Fransızca
    'de': 'normale Augenbilder',   # Almanca
    'es': 'imágenes normales del ojo',   # İspanyolca
    'it': 'immagini normali dell occhio',  # İtalyanca
    'pt': 'imagens normais do olho',   # Portekizce
    'ru': 'нормальные изображения глаза',  # Rusça
    'ar': 'صور العين الطبيعية',        # Arapça
    'ja': '通常の目の画像',       # Japonca
    'ko': '정상 눈 이미지',       # Korece
    'zh-CN': '正常眼睛图片',    # Çince (Basitleştirilmiş)
    'zh-TW': '正常眼睛圖片',    # Çince (Geleneksel)
    'hi': 'सामान्य आंख की छवियाँ', # Hintçe
    'bn': 'নরমাল চোখ ছবি',    # Bengalce
    'ur': 'عام آنکھ کی تصاویر', # Urduca
    'ta': 'சாதாரண கண் படங்கள்', # Tamilce
    'te': 'సాధారణ నేత్రం చిత్రాలు', # Teluguca
    'sq': 'fotografi të zakonshme të syrit',   # Arnavutça
    'bg': 'нормални снимки на око',    # Bulgarca
    'hr': 'normalne slike oka',   # Hırvatça
    'da': 'normale øjebilleder',    # Danca
    'fi': 'normaalit silmäkuvat',  # Fince
    'el': 'κανονικές φωτογραφίες ματιού',  # Yunanca
    'hu': 'normál szem képek', # Macarca
    'no': 'normale øye bilder'     # Norveççe
}


folder = 'normal_images'  # Kaydedilecek klasör

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
            #with open(os.path.join(folder, f'{lang_code}_{i}.jpg'), 'wb') as f:
                #f.write(img_data)
    
    #print(f"{len(images[:100])} images downloaded for {lang_code}.")

# Tarayıcıyı kapat
driver.quit()


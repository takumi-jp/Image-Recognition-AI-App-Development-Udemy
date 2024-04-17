import os
import sys
import time
from pprint import pprint
from urllib.request import urlretrieve

from dotenv import load_dotenv
from flickrapi import FlickrAPI
from tqdm import tqdm

# API Key Info
load_dotenv("./.env")
KEY = os.getenv("API_KEY")
SECRET = os.getenv("SECRET_KEY")
wait_time = 1

# Save fike
animal_name = sys.argv[1]
save_dir = "../data/" + animal_name

flickr = FlickrAPI(KEY, SECRET, format="parsed-json")
result = flickr.photos.search(
    text=animal_name,  # 検索キーワードを指定
    per_page=1,  # 取得枚数を指定
    media='photos',  # データの種類を指定
    sort='relevance',  # 最新から取得
    safe_search=1,  # 暴力的な画像は対象外にする
    extras='url_q, license'  # URLとライセンス情報を取得する
)
photos = result['photos']  # 写真データ部分を取り出す
# pprint(photos)

for i, photo in tqdm(enumerate(photos['photo']), desc='Downloading photos', total=len(photos['photo'])):
    url_q = photo['url_q']
    file_path = save_dir + '/' + photo['id'] + '.jpg'

    if os.path.exists(file_path):
        continue
    urlretrieve(url_q, file_path)
    time.sleep(wait_time)

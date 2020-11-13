import requests
from bs4 import BeautifulSoup as bs
import os
import pandas as pd
import ssl

def is_meta_description(tag):
    return tag.name == 'meta' and tag['name'] == 'thumbnail'
df = pd.read_csv("data.csv")
save_path='./memes/'
for i in range(1000):
    print(i)
    img_uid = df['img_id'][i]
    url = 'https://memegenerator.net/instance/'+str(img_uid)
    r = requests.get(url)
    soup = bs(r.text,'html.parser')
    image_url = soup.select_one('[rel="image_src"]')['href']
  
    complete_name = os.path.join(save_path, f"{img_uid}.jpg")
    response = requests.get(image_url, stream=True)
  
    with open(complete_name, 'wb') as im_file:
        im_file.write(response.content)
    del response
  
  


meta_tag = soup.find(is_meta_description)
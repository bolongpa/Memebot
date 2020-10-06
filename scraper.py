import requests
from bs4 import BeautifulSoup as bs
import os
import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

n_pages=3
n_captions=15
save_path='./memes/'

df=pd.DataFrame(columns=['img_id','caption'])

# http://memegenerator.net/images/popular/alltime/page/1

for i in range(1,n_pages+1):
  url = 'https://memegenerator.net/images/popular/alltime/page/'+str(i)
  print("page_all:",str(i))

  r = requests.get(url)
  soup = bs(r.text,'html.parser')
  memes = soup.find_all(class_='generator-img')
  base_memes = soup.find_all(class_='generator-meta')
  imgs = [meme.find('img') for meme in memes]
  links = [meme.find('a') for meme in memes]
  base_imgs = [bm.find('a') for bm in base_memes]

  for j, img in enumerate(imgs):
    img_url = img['src']
    img_name = base_imgs[j]['href'].split('/')[-1].lower()
    response = requests.get(img_url, stream=True)
    name = links[j]['href'].split('/')[-1].lower()
    caption = name.replace(f"{img_name}-", "", 1)
    img_uid = links[j]['href'].split('/')[-2].lower()
    complete_name = os.path.join(save_path, f"{img_uid}.jpg")
    print(f"Saving {img_uid} to {complete_name}: base image {img_name}...")
    with open(complete_name, 'wb') as im_file:
      im_file.write(response.content)
    del response
    df = df.append({'img_id': img_uid, 'caption': " ".join(caption.split("-")), 'base_img_name': img_name, 'base_img_url': f"http://memegenerator.net/{base_imgs[j]['href']}"},
                  ignore_index=True)
        
df.to_csv('data.csv')

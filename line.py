import requests
import json
url = 'https://notify-api.line.me/api/notify'
# read config.json to get the token
with open('config.json') as config_file:
    config = json.load(config_file)
    token = config['token']
    
def lineNotify(msg,img,pid,sid):
    image = open('./output.jpg', 'rb') 
    imageFile = {'imageFile' : image}   # 設定圖片資訊
    headers = {
        'Authorization': 'Bearer ' + token,
    }
    data = {
        'message':msg,
        'stickerPackageId':pid,
        'stickerId':sid,
    }
    f = requests.post(url, headers=headers, data=data,files=imageFile)

if __name__ == '__main__':
    lineNotify('lineNotify test!','./output.jpg','6632','11825389')

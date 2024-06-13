import requests

url = 'https://notify-api.line.me/api/notify'
token = 'qlClFGFwH1QNAZtRaIWLfexbv0LNmsuLQ0CVu7a32yK'
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
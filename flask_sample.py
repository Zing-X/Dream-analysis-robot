from flask import Flask, request

# 載入 json 標準函式庫，處理回傳的資料格式
import json

# 載入 LINE Message API 相關函式庫
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

#from chatbot_sample_training import tokenize_chinese, fit_sentence, build_model
from train_chatbot_GRU import tokenize_chinese, fit_sentence, build_model
#from train_chatbot_bert import tokenize_chinese, fit_sentence, build_model
from chatbot_sample_inference import load_model, single_predict

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def linebot():
    body = request.get_data(as_text=True)                    # 取得收到的訊息內容
    try:
        json_data = json.loads(body)                         # json 格式化訊息內容
        access_token = 'tuJAoBXbKSWA35Wd3S/+Ec8s9M31xweXjldeVU7Or69BWYwuWFkKBFxINinSlPZ5aJyHwBpKTjVHf/89K/m7Idqb0ZA0/Zy4zr1AgTHPsa6Uy8MgkiOf1sdjXFj5/CfrarrRe568FWoxe9yMA5wz5gdB04t89/1O/w1cDnyilFU='
        secret = '75e797966605e208e68ddc3251eaf3ef'
        line_bot_api = LineBotApi(access_token)              # 確認 token 是否正確
        handler = WebhookHandler(secret)                     # 確認 secret 是否正確
        signature = request.headers['X-Line-Signature']      # 加入回傳的 headers
        handler.handle(body, signature)                      # 綁定訊息回傳的相關資訊
        tk = json_data['events'][0]['replyToken']            # 取得回傳訊息的 Token
        receive_type = json_data['events'][0]['message']['type']     # 取得 LINE 收到的訊息類型
        if receive_type == 'text':
            msg = json_data['events'][0]['message']['text']  # 取得 LINE 收到的文字訊息
            print(f'User: {msg}')   
            voc, ind_voc, model = load_model()
            if msg == '88':
                res = '再見'
            elif msg == '亂講':
                res = '對不起，我不該亂講話'
            else:
                res = single_predict(msg, voc, ind_voc, model)                            
            reply = res                                      # Change your reply into your predict result of chatbot model
        else:
            reply = '請傳送文字'
        print(reply)
        line_bot_api.reply_message(tk,TextSendMessage(reply))# 回傳訊息
    except:
        print(body)                                          # 如果發生錯誤，印出收到的內容
    return 'OK'                                              # 驗證 Webhook 使用，不能省略

if __name__ == "__main__":
    app.run()
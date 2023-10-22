# Dream-analysis-robot

## 動機
有一個有趣的朋友總是做一些奇怪且難以理解的夢。他曾告訴我，有一次他夢見自己開著教授的車去拜訪外婆。對於他的這奇怪夢境，我們是否可以使用AI模型來分析並解釋它們的意義呢？
因次透過這次自然語言處理課程的期末專案，我訓練了一個可以解析夢境的聊天機器人模型，並使用ngrok及flask串接到Line Bot上方便使用

## 資料集
透過爬蟲蒐集一些解夢網站及論壇上的夢境分析，加上其他人做過類似的資料集加以處理，一共3萬多筆資料

## 模型
1. encoder和decoder為LSTM
2. encoder和decoder為GRU
3. encoder為bert，decoder為LSTM(未練完)

## GRU的訓練成果
![image](https://github.com/Zing-X/Dream-analysis-robot/assets/135576414/a10d490e-859a-427c-b069-66b9dd4edfe2)
![image](https://github.com/Zing-X/Dream-analysis-robot/assets/135576414/f585086d-e967-40fa-b42f-2df85738335f)

## Line Bot
![image](https://github.com/Zing-X/Dream-analysis-robot/assets/135576414/aa0b61f9-f548-4126-8d37-b7967132ddb3)
![image](https://github.com/Zing-X/Dream-analysis-robot/assets/135576414/2de3f96d-2db2-48fb-9f55-161c7753f26f)
![image](https://github.com/Zing-X/Dream-analysis-robot/assets/135576414/cbc82762-89a0-447c-8320-b3e1ece1ed9d)

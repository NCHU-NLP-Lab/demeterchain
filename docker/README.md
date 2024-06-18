# demeterchain
demeterchain的簡易測試環境
## 啟動
在此資料夾輸入以下指令建立container  
```=
model=NchuNLP/taide-qa
token=<your huggingface hub token>
docker run --gpus all \
	--rm \
	-p 8000:8000 \
	-e HUGGING_FACE_HUB_TOKEN=$token \
	-v ./data:/data \
	-v ./model:/model \
	nchunlplab/demeterchain:1.0.2 \
    	--model_name_or_path $model \
        --device_map auto
```
啟動後會讀取./data底下的文章並建立retriever  
使用的模型會儲存在./model  
也可以先複製模型到此資料夾
## 參數
+ `--port` : 預設為8000，api開啟的port
+ `--model_name_or_path` : 預設為NchuNLP/taide-qa，本地路徑或huggingface上模型的路徑
+ `--device_map` : 預設為auto，模型要放在甚麼裝置
+ `--dtype` : 預設為float16，模型讀取的型態，可使用float32, float16, bfloat16
+ `--quantize` : 預設為None，量化模型，提供以下兩種選擇
    - `bitsandbytes` : 等同於load_in_8bit
    - `bitsandbytes-nf4` : 等同於load_in_4bit並使用nf4
+ `--use_flash_attention` : 預設為False，是否啟用flash_attention_2
+ `--retrieve_k` : 預設為3，檢索的文章數量
## 測試
透過以下code進行測試
```=
import requests

url = "http://127.0.0.1:8000/qa"
data = {
    "query": "甚麼水果產於南美洲?"
}
response = requests.post(url, json=data)

answers = response.json()
for answer in  answers:
    print(answer)
```
輸出結果
```=
{'answer': '鳳梨', 'doc': '鳳梨（Ananas comosus），俗名菠蘿、露兜子[1]，是原產於南美洲的熱帶水果，為禾本目鳳梨科鳳梨屬植物，因多汁酸甜受到喜愛，有解暑之效，是鳳梨科中最具經濟價值的種類。南美洲種植鳳梨已有許多世紀，在17世紀傳入歐洲，於1820年代開始在溫室與熱帶地區開始商業種植。在20世紀，夏威夷是重要的鳳梨產地（尤以美國而言），但到了2016年， 哥斯大黎加、巴西、菲律賓合計佔了世界鳳梨產量將近三分之一。[2]'}
```
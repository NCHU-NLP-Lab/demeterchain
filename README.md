# demeterchain
透過retriever檢索文章  
並從文章中萃取答案
## Installation
Install via PyPI
```=
pip install demeterchain
```
為了避免下載過多無用套件  
使用時須根據自己的需求自行安裝其他套件  

## 流程簡介
### 建立retriever流程
``` mermaid
graph TD;
讀取本地文件-->分割文件;
分割文件-->建立retriever;
```

### 檢索流程
``` mermaid
graph TD;
使用者輸入問題-->Hyde["(可省略)使用HyDE擴增問題"];
Hyde-->檢索相關文章;
檢索相關文章-->從每篇檢索到的文章尋找答案;
從每篇檢索到的文章尋找答案-->Summary["(可省略)摘要並統整所有答案"];
Summary-->回傳結果;
```

## 使用說明
使用[examples/demo.ipynb](examples/demo.ipynb)進行簡單測試  
使用[examples/complete_demo.ipynb](examples/complete_demo.ipynb)進行完整測試  

## 功能介紹
此處並不會介紹全部功能  
僅針對部分功能進行介紹  

### TextSplitter
將文檔進行分割  
+ `separator` : 分割文檔時只能在遇到separator才進行分割，若不設定則會以長度進行分割
+ `chunk_size` : 預期的分割文檔長度，當使用separator時可能會有比chunk_size長或短的結果出現
+ `chunk_overlap` : 分割文檔之間重疊的長度

### PyseriniBM25Retriever
使用Pyserini的bm25為基底的retriever  
需安裝jdk11  
```=
sudo apt-get update
sudo apt-get install openjdk-11-jdk
```
與Pyserini, faiss-cpu
```=
pip install pyserini==0.22.1 faiss-cpu==1.7.4
```

### RankBM25Retriever
使用rank_bm25的BM25Okapi為基底的retriever  
需安裝rank_bm25 
```=
pip install rank_bm25
```

### QAModelConfig
設定讀取模型時的各種參數  
+ `model_name_or_path` : str，本地路徑或huggingface上模型的路徑，建議使用"NchuNLP/taide-qa"
+ `template` : 建議直接參考[examples/demo.ipynb](examples/demo.ipynb)
+ `device_map` : str，模型要放在甚麼裝置
+ `dtype` : str，模型讀取的型態，可使用float32, float16, bfloat16
+ `quantize` : str，量化模型，提供以下兩種選擇
    - `bitsandbytes` : 等同於load_in_8bit
    - `bitsandbytes-nf4` : 等同於load_in_4bit並使用nf4
+ `use_flash_attention` : bool，是否啟用flash_attention_2，[安裝方式](https://huggingface.co/docs/transformers/perf_infer_gpu_one)
+ `noanswer_str` : 建議直接參考[examples/demo.ipynb](examples/demo.ipynb)
+ `noanswer_ids` : 建議直接參考[examples/demo.ipynb](examples/demo.ipynb)

### QAConfig
設定檢索及回答問題時的各種參數  
+ `retrieve_k` : int，retriever檢索的篇數
+ `batch_size` : int，模型同時處理的文章數量，請依照自身顯卡vram進行調整
+ `max_length` : int，模型所能接受的最大長度，請依照自身顯卡vram進行調整，預設為768
+ `max_new_tokens` : int，模型預測的最大長度
+ `num_beams` : int, 生成過程中的答案數量，用來提升解碼的精準度
+ `answer_strategy` : 如何決定一篇文章的答案
    - `best` : 模型預測的最佳結果
    - `longest` : 模型產生的答案中最長的一個
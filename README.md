# item recommendation using i2i deepwalk.
簡單的實作 ^ ^

Reference: Wang, Jizhe, et al. "Billion-scale commodity embedding for e-commerce recommendation in alibaba." Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.

## 環境
* python 3.6.10
* pyspark==2.4.6
* gensim==3.8.3
* faiss-cpu==1.6.5

## 建立 i2i index:
(請移步至 `i2i/` 執行)
1. 執行 `generate_item_outdegree.py`
    * 整理資料，計算所有 item 的 next-item count (out degree)
    * 根據 link prediction 情境， 刪除部分 out-degree ，並生成負樣本作為 validatoin set.
2. 執行 `generate_deepwalk_sentence.py`
    * 抽樣 deepwalk 用 seqeunce
3. 執行 `generate_i2i_index.py`
    * 訓練 word2vec embedding
    * 利用 faiss 建立 item 推薦清單



## Serving:
* Use Flask:
    * Build (在專案目錄下):
    ```
    docker build -t i2i-flask -f docker/flask/. .
    ```
    * Run container:
    ```
    docker run --name i2i-demo-flask -p 8500:80 i2i-flask
    ```
    * request example (in python):
    ```
    import requests
    import json

    req = json.dumps({'item':'11', 'k':10})
    res = requests.post('http://localhost:8500/predict', json=req)
    ```

* Use Redis:
    * Build (在專案目錄下):
    ```
    docker build -t i2i-redis  -f docker/redis/. .
    ```
    * Run container
    ```
    docker run --name i2i-demo-redis -itd -p 6379:6379 i2i-redis
    ```
    * connect from local machine
    ```
    redis-cli -h 127.0.0.1 -p 6379
    ```
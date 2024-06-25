#  Funi

## 運行Funi
### 前置作業
1. 下載 requirement.txt 中的所有 python 庫
2. 更改 funi.py 中的變數 `model_path` 到您的llm目錄，以運行大語言模型
3. 新增 .env 文件

* 若要下載大語言模型，請前往 [Hunggingface](https://huggingface.co/) ，選擇 [Models](https://huggingface.co/models) ，並選擇適合的模型
* 若要連線到 Discord ，請在 .env 文件中新增 `token = <your_toekn>`

### 運行模型
1. 運行 local.py 或 connect_to_discord.py ，運行大語言模型

## 個人化、微調模型
### 前置作業
1. 在 .env 文件中新增 `origin_model = <model_want_to_train>` 、 `fine_tuned_model = <trained_model_output_dir>`
2. 檢視資料集中是否有不需要的訓練資料
3. 調整 fine_tune.py 中的 training_loop 到適合的次數

### 使用方法
1. 客製化 dataset.py ，透過修改變數將資料集調整成自己想要的樣子
2. 運行 fine_tune.py


## 備註
* 自述文件尚未完成，有很多內容沒提到
* 有些資料和變數可能因為不存在於使用者的環境，從而造成錯誤
* dataset.py 中的資料量尚未擴充，可能不夠充分
* online_search.py 尚未完成和實裝
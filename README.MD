# Linear Regression 機器學習專案說明

## 專案概述

這是一個簡單的機器學習應用，使用scikit-learn的線性迴歸模型來預測學生根據讀書時間得到的考試分數，並提供Gradio網頁界面進行互動式預測。

## 目錄結構及檔案說明

/
├── app_linear_regression.py    # Gradio應用程式主文件
├── docker-compose.yaml         # Docker容器編排配置
├── Dockerfile                  # Docker容器建構文件
├── Dockerfile.ERR              # 有語法錯誤的Docker配置檔案
├── models/                     # 儲存訓練好的模型
│   └── linear_regression_model.joblib  # 序列化的線性迴歸模型
└── notebooks/                  # Jupyter筆記本目錄
    ├── docker-compose.yml      # 筆記本專用的Docker編排配置
    ├── Dockerfile.orig         # 原始Docker配置
    ├── train_linear_regression.ipynb  # 主要模型訓練筆記本
    ├── train_linear_regression1.ipynb # 訓練筆記本的替代版本
    ├── Untitled.ipynb          # 未命名筆記本
    └── .ipynb_checkpoints/     # Jupyter自動生成的檢查點備份
## 主要功能模組

### 1. 模型訓練 (`notebooks/train_linear_regression.ipynb`)

模型訓練筆記本包含以下步驟：

1. **數據模擬生成**：
   - 生成100個樣本點，讀書時間(1-20小時)與對應的考試分數(0-100分)
   - 使用公式：考試分數 = 10 + 4×讀書時間 + 隨機噪音
   - 使用numpy生成隨機數並添加合理的噪聲

2. **資料視覺化**：
   - 繪製讀書時間與考試分數的散點圖

3. **資料分割**：
   - 80%訓練集，20%測試集
   - 使用`train_test_split`分割數據

4. **模型訓練**：
   - 使用scikit-learn的`LinearRegression`模型
   - 訓練結果：斜率約為3.83，截距約為11.31

5. **模型評估**：
   - 均方誤差(MSE)：約41.84
   - R平方(R²)：約0.92
   - 視覺化預測結果與實際數據的比較

6. **模型儲存**：
   - 使用joblib將模型序列化儲存到Docker容器中的`/app/models/linear_regression_model.joblib`路徑

### 2. 預測應用 (`app_linear_regression.py`)

Gradio Web應用程式提供以下功能：

1. **模型載入**：
   - 從容器路徑加載訓練好的模型
   - 檢查並顯示模型參數(斜率和截距)

2. **預測函數**：
   - 接收用戶輸入的讀書時間
   - 使用模型預測考試分數
   - 結果限制在0-100分範圍
   - 包含錯誤處理機制

3. **Web界面**：
   - 使用Gradio創建簡潔的UI界面
   - 包含輸入框、輸出顯示和預設範例
   - 運行在7861端口

### 3. Docker配置

專案使用Docker容器化以確保環境一致性：

1. **基本環境**：
   - 基於Python 3.9-slim鏡像
   - 安裝必要套件：jupyterlab、gradio、numpy、joblib、scikit-learn、pandas、matplotlib

2. **端口映射**：
   - Jupyter服務：8888(容器) → 8890(主機)
   - Gradio服務：7861(容器) → 7861(主機)

3. **目錄掛載**：
   - 將本地專案目錄掛載到容器的`/app`目錄

## 使用流程

1. 啟動Docker容器：
   ```
   docker-compose up -d
   ```

2. 運行Jupyter筆記本訓練模型：
   - 訪問`http://localhost:8890`
   - 打開並執行`notebooks/train_linear_regression.ipynb`
   - 訓練完成後模型會儲存到指定位置

3. 啟動Gradio應用：
   ```
   python app_linear_regression.py
   ```
   或在容器中運行該腳本

4. 使用Web界面進行預測：
   - 瀏覽器訪問`http://localhost:7861`
   - 輸入讀書時間，查看預測的考試分數

## 技術特點

1. 模型關係簡單直觀：每學習一小時大約增加3.83分，基礎分數為11.31分
2. 資料生成具有隨機性，模擬了實際學習效果的波動
3. Docker容器化確保環境一致性，避免"在我電腦上能運行"的問題
4. Gradio提供簡潔的互動式界面，方便非技術人員使用

這個專案是機器學習模型訓練和部署的簡單範例，展示了從資料生成、模型訓練到應用部署的完整流程。

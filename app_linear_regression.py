# linear_regression_sklearn/app_linear_regression.py
import gradio as gr
import joblib
import numpy as np
import os

# --- 設定 ---
MODEL_FILENAME = 'linear_regression_model.joblib'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME) # -> models/linear...joblib

PORT_NUMBER = 7861 # Gradio 端口
# --- ---

container_model_path = os.path.join("/app", MODEL_PATH) # -> /app/models/linear...joblib
print(f"嘗試從以下路徑載入模型: {container_model_path}")

# --- 載入模型 ---
model = None
if not os.path.exists(container_model_path):
    print(f"錯誤: 在 '{container_model_path}' 找不到模型檔案。")
    print("請先運行 'notebooks/train_linear_regression.ipynb' 來訓練並儲存模型。")
else:
    print("正在載入模型...")
    try:
        model = joblib.load(container_model_path)
        print("Scikit-learn 線性迴歸模型載入成功。")
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
             print(f"  - 模型斜率: {model.coef_[0]:.4f}, 截距: {model.intercept_:.4f}")
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        model = None

# --- 定義推論函數 ---
def predict_exam_score_gradio(study_hours):
    if model is None:
        return "錯誤：模型未能載入。"
    if study_hours is None:
        return None # 或者返回 0 或提示
    try:
        input_data = np.array([[float(study_hours)]])
        predicted_score = model.predict(input_data)
        final_score = np.clip(predicted_score[0], 0, 100) # 限制 0-100
        return round(final_score, 2) # 返回數字
    except ValueError:
         return "錯誤：請輸入有效的數字。"
    except Exception as e:
        print(f"預測時發生錯誤: {e}")
        return f"預測錯誤: {e}"

# --- 建立 Gradio 介面 ---
print(f"正在啟動 Gradio 介面於連接埠 {PORT_NUMBER}...")

interface = gr.Interface(
    fn=predict_exam_score_gradio,
    inputs=gr.Number(label="請輸入讀書時間 (小時)"),
    outputs=gr.Number(label="預測分數 (0-100)"),
    title="讀書時間預測考試分數 (Scikit-learn)",
    description=f"使用 Scikit-learn 線性迴歸模型進行預測。模型來源: /app/{MODEL_PATH}",
    examples=[[0.5], [3.0], [8.0], [15.0]],
    allow_flagging='never'
)

# --- 啟動 Gradio Web 伺服器 ---
interface.launch(server_name="0.0.0.0", server_port=PORT_NUMBER)

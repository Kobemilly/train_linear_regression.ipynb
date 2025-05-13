# linear_regression_sklearn/Dockerfile
FROM python:3.9-slim
WORKDIR /app
# 更新 pip 並安裝必要的 Python 套件
RUN pip install --no-cache-dir --upgrade pip && \
    pip install \
        --no-cache-dir \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        jupyterlab gradio numpy joblib scikit-learn pandas matplotlib
# 宣告端口...
EXPOSE 8888
EXPOSE 7861
CMD ["tail", "-f", "/dev/null"]

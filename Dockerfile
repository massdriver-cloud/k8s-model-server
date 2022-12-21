FROM gcr.io/deeplearning-platform-release/pytorch-gpu

RUN pip install \
  Flask flask-cors \
  diffusers transformers accelerate

COPY . .

CMD ["python", "main.py"]

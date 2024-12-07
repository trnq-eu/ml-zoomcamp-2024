FROM public.ecr.aws/lambda/python:3.8.2024.11.22.14

RUN pip install keras-image-helper
RUN pip install pip install --extra-index-url \
    https://google-coral.github.io/py-repo/ tflite_runtime

COPY clothin-model.tflite .
COPY lambda_function.py .  

CMD [ "lambda_function.lambda_handler"]
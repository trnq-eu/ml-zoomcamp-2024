# Use Python 3.12 slim
FROM python:3.12-slim

# Create working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy pipenv files into the working directory
COPY ["Pipfile", "Pipfile.lock", "xgbm_final_model_local.pkl", "./"]

# run pipenv and deploy
RUN pipenv install --system --deploy

# copy Flask file
COPY ["predict.py", "./"]

# Expose the port
EXPOSE 9696

# Entrypoints
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"] 
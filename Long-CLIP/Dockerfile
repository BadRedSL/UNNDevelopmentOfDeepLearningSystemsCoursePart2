FROM python:3.9

COPY . /app

VOLUME /app_volume

WORKDIR /app

RUN pip install -r ./requirements.txt

CMD ["python", "docker_demo.py"]
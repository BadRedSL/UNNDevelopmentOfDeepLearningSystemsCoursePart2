FROM python:3.9

COPY . /app

WORKDIR /app

VOLUME /app_volume

RUN pip install -r ./requirements.txt

CMD ["python", "docker_demo.py"]
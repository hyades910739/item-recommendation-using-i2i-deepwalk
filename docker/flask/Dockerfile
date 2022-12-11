FROM tiangolo/uwsgi-nginx-flask:python3.8

ENV UWSGI_CHEAPER 4

ENV UWSGI_PROCESSES 16

ENV I2I_FILENAME i2i_map.json

RUN apt-get update

COPY i2i_serving /app

COPY files/i2i_map.json /app

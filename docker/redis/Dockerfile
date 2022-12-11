FROM redis:latest
RUN mkdir /init/
COPY files/i2i.redis /init/
COPY docker/redis/start-redis.sh /init/
CMD ["sh", "/init/start-redis.sh"]

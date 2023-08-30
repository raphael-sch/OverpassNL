FROM wiktorn/overpass-api:0.7.57.2
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker-entrypoint.sh /app/

HEALTHCHECK --interval=259200s CMD echo "Dummy health check"

CMD ["/app/docker-entrypoint.sh"]
FROM apache/airflow:2.10.5
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /sources/data/{raw,processed,monitoring,features} \
    && mkdir -p /sources/data/processed/{clean,engineered} \
    && mkdir -p /opt/airflow/logs \
    && chown -R airflow:root /sources /opt/airflow/logs \
    && chmod -R 775 /sources /opt/airflow/logs
USER airflow
RUN echo -e "AIRFLOW_UID=$(id -u)" > .env
COPY --chown=airflow:root requirements.txt /
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt

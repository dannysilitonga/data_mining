FROM python:3.8.9
WORKDIR /data_mining
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . /data_mining
ENTRYPOINT [ "streamlit", "run" ]
CMD ["predict_stocks.py"]


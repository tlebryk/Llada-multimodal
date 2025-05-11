FROM python:3.12-slim 

WORKDIR /app

RUN apt-get update && apt-get install -y zip unzip wget
# RUN 
RUN wget -O cat.jpg https://cataas.com/cat
# RUN 
RUN wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.zip \
&& for i in $(seq -f "%02g" 1 9); do \
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z$i; \
done \
&& zip -F pix2code_datasets.zip --out datasets.zip \
&& unzip datasets.zip -d ./datasets/COCO/ 

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["python", "llada_train.py"]



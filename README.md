# DE2025

How to run the files

cd prediction-ui
sudo docker build -t prediction-ui:0.0.1 .
sudo docker run -p 5001:5000 -d --name=prediction-ui prediction-ui:0.0.1

cd ..
cd prediction-api
sudo docker build -t prediction-api:0.0.1 .
sudo docker run -p 5000:5000 -d --name=prediction-api prediction-api:0.0.1

sudo docker network create house-predictor-network
sudo docker network connect house-predictor-network prediction-api
sudo docker network connect house-predictor-network prediction-ui

sudo docker build -t prediction-ui:0.0.1 .
sudo docker rm -f prediction-ui || true
sudo docker run -p 5001:5000 -d --name=prediction-ui prediction-ui:0.0.1
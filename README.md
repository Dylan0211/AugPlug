# AugPlug: An Automated Data Augmentation Model to Enhance Online Building Load Forecasting


## Tested Online BLF Models: 
- SVM (Periodically Retrain): Designed an LF strategy based on support vector regression machines (SVR) with a feature selection algorithm. The data collected within the recent 30 days is leveraged to retrain an SVM-based energy load prediction model weekly, to adapt to the pattern shift during the COVID-19 pandemic in Spain. This online LF belongs to Periodically + Retrain. 


- LSTM (Periodically Fine-tune): Designed an adaptive buffer to keep the useful and difficult previous data samples for updating the LSTM-based forecasting model. while in use, the LSTM model will be periodically (every ten, 20, or 40 days) fine-tuned on the recorded data in the buffer. This LF model belongs to Periodically + Fine-tune. 


- RF (Triggered Retrain): proposed a two-stage K-Nearest Neighbors(KNN) load forecasting model which based on i) data transformation and ii) an error-based updating mechanism. While in use, the KNN will be retrained using the last 20 working days when the error exceeds the defined threshold (e.g. 0.25 in Weight Absolute Percentage Error (WAPE)). This LF model belongs to Triggered + Retrain.


- Autoencoder (Triggered Fine-tune): This paper proposes an LF method based on ensemble learning with online adaptive Recurrent Neutal Network(RNN) and ARIMA, it designed an adaptive window (the window size is dynamic) to select suitable historical data as the input. In the deployment process, the prediction error of the last windows is recorded and leveraged to fine-tune the RNN model. This belongs to Triggered + Fine-tune.


## How to apply AugPlug to support online BLF Models:

<img width="433" alt="augplug_1" src="https://github.com/user-attachments/assets/b7cc0629-2c2c-43a4-865d-d42aa6d123ca">

<img width="461" alt="augplug_2" src="https://github.com/user-attachments/assets/7244a9cd-67a8-4663-a23a-249645983737">


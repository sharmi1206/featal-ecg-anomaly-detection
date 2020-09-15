Reference Blog : https://techairesearch.com/anomaly-detection-from-head-and-abdominal-fetal-ecg-a-case-study-of-iot-anomaly-detection-using-generative-adversarial-networks/

Reference Paper : Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications – https://arxiv.org/abs/1802.03903

# fetal-ecg-anomaly-detection
Fetus Anomaly detection using Variational Auto Encoder

https://physionet.org/content/adfecgdb/1.0.0/ -Dataset used for modeling

Since the data format is in edf, it needs to be converted to csv, which is done with the mne library

fetalAnomalyAbdomen.py - Determines fetal heart rate anomaly from electrocardiogram signals captured from maternal abdomen
fetalAnomalyBrain.py - Identifies anomaly from electrocardiogram signals captured from fetal brain

Pre-requisites :

 pip install donut
 
 pip install tfsnippet
 
 pip install mne
 
Since data was initially not labelled, the initial labelling of anomalous electrocardiogram signals was done by considering signals >= .00025 and signals <= -.00025. However the same could be accomplished by using a single class SVM.




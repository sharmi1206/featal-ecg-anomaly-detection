import numpy as np
from donut import complete_timestamp, standardize_kpi
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11, 4)})
from sklearn.metrics import accuracy_score
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_path = '/Users/shachatt1/PycharmProjects/IOTSimulation/featal-ecg-anomaly-detection/abdominal-and-direct-fetal-ecg-database-1.0.0/'
file_name = 'r10.edf'
save_dir ='/Users/shachatt1/PycharmProjects/IOTSimulation/featal-ecg-anomaly-detection/model_dir/abdomen/'

edf = mne.io.read_raw_edf(data_path+file_name)
header = ','.join(edf.ch_names)
np.savetxt('r10.csv', edf.get_data().T, delimiter=',', header=header)

df = pd.read_csv('r10.csv')
periods = df.shape[0]

dti = pd.date_range('2018-01-01', periods=periods, freq='s')
print(dti.shape, df.shape)
df['DateTs'] = dti


df.set_index('DateTs')
df.index = pd.to_datetime(df.index, unit='s')
df1 = df.resample('1T').mean()

#df1['timestamp'] = df1.index
print(df1.shape)


cols = df1.columns
df1.rename_axis('timestamp', inplace=True)
print(cols, df1.index.name)

df1['label'] =  np.where((df1['Abdomen_1'] >= .00025) | (df1['Abdomen_1'] <= -.00025), 1, 0)
print(df1.head(5))

for i in range(0, len(cols)):
    if(cols[i] != 'timestamp'):
        plt.figure(figsize=(20, 10))
        plt.plot(df1[cols[i]], marker='^', color='red')
        plt.title(cols[i])
        plt.savefig('figs/f_' + str(i) + '.png')


df2 = df1.reset_index()
df2 = df2.reset_index(drop=True)


# Read the raw data for 1st feature Direct_1/2nd feature Abdomen_1
timestamp, values, labels = df2['timestamp'], df2['Abdomen_1'], df2['label']
# If there is no label, simply use all zeros.
labels = np.zeros_like(values, dtype=np.int32)


# Complete the timestamp, and obtain the missing point indicators.
timestamp, missing, (values, labels) = \
    complete_timestamp(timestamp, (values, labels))



# Split the training and testing data.
test_portion = 0.3
test_n = int(len(values) * test_portion)
train_values, test_values = values[:-test_n], values[-test_n:]
train_labels, test_labels = labels[:-test_n], labels[-test_n:]
train_missing, test_missing = missing[:-test_n], missing[-test_n:]

# Standardize the training and testing data.
train_values, mean, std = standardize_kpi(
    train_values, excludes=np.logical_or(train_labels, train_missing))
test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)


import tensorflow as tf
from donut import Donut
from tensorflow import keras as K
from tfsnippet.modules import Sequential

# We build the entire model within the scope of `model_vs`,
# it should hold exactly all the variables of `model`, including
# the variables created by Keras layers.
with tf.variable_scope('model') as model_vs:
    model = Donut(
        h_for_p_x=Sequential([
            K.layers.Dense(50, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(50, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),

        ]),
        h_for_q_z=Sequential([
            K.layers.Dense(50, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(50, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        x_dims=120,
        z_dims=5,
    )


from donut import DonutTrainer, DonutPredictor

trainer = DonutTrainer(model=model, model_vs=model_vs, max_epoch=512)
predictor = DonutPredictor(model)

with tf.Session().as_default():
    trainer.fit(train_values, train_labels, train_missing, mean, std)
    test_score = predictor.get_score(test_values, test_missing)

    pred_score = np.array(test_score).reshape(-1, 1)
    print(len(test_missing), len(train_missing), len(pred_score), len(test_values))
    y_pred = np.argmax(pred_score, axis=1)


    plt.figure(figsize=(20, 10))
    split_test  = int((test_portion)*df.shape[0])

    anomaly = np.where(pred_score > -3, 0, 1)
    print("Anomaly shape", anomaly.shape, anomaly)

    df3 = df2.iloc[-anomaly.shape[0]:]
    df3['outlier'] = anomaly
    df3.reset_index(drop=True)

    print(df3.head(2), df3.shape)
    print("Split", split_test, df3.shape)
    di = df3[df3['outlier'] == 0]
    do = df3[df3['outlier'] == 1]

    di = di.set_index(['timestamp'])
    do = do.set_index(['timestamp'])

    print("Outlier and Inlier Numbers", do.shape, di.shape, di.columns, do.columns)
    print(di.head(5))
    print(do.head(5))

    outliers = pd.Series(do['Abdomen_1'], do.index)
    inliers = pd.Series(di['Abdomen_1'], di.index)

    plt.plot(do['Abdomen_1'], marker='^', color='red', label="Anomalies")
    plt.plot(di['Abdomen_1'],  marker='^', color='green', label="Non Anomalies")

    plt.legend(['Anomalies', 'Non Anomalies'])
    plt.title('Anomalies and Non Anomalies from Fetal Head Scan')
    plt.savefig('figs/out_anomaly_Abdomen_1.png')

    di = di.reset_index()
    do = do.reset_index()
    plt.figure(figsize=(20, 10))

    do.plot.scatter(y ='Abdomen_1', x = 'timestamp', marker='^', color='red', label="Anomalies")

    plt.legend(['Anomalies'])
    plt.xlim(df3['timestamp'].min(), df3['timestamp'].max())
    plt.title('Anomalies from Fetal Abdomen Scan')
    plt.ylim(-.00006, .00006)
    plt.savefig('figs/out_abd1_anomaly.png')


    plt.figure(figsize=(20, 10))
    di.plot.scatter(y='Abdomen_1', x='timestamp', marker='^', color='green', label="Non Anomalies")
    plt.legend(['Non Anomalies'])
    plt.xlim(df3['timestamp'].min(), df3['timestamp'].max())
    plt.ylim(-.00006, .00006)
    plt.title('Non Anomalies from Fetal Abdomen Scan')
    plt.savefig('figs/out_abd1_nanomaly.png')



from tfsnippet.utils import get_variables_as_dict, VariableSaver


session = K.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)

with session.as_default():

    var_dict = get_variables_as_dict(model_vs)
    # save variables to `save_dir`
    saver = VariableSaver(var_dict, save_dir)
    saver.save()
    print("Saved the model successfully")

with session.as_default():
    # Restore the model.
    saver = VariableSaver(get_variables_as_dict(model_vs), save_dir)
    saver.restore()
    print("Restored the model successfully")







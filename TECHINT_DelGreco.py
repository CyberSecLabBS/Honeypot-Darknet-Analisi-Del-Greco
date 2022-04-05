import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import Sequential
from keras.layers import Dense
from keras import callbacks
from clusteval import clusteval
from sklearn import metrics
import numpy as np


numeric_cols = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'pck_len', 'tcp_flags', '1_port', 'port_list', 'port_list_no']
data = pd.read_csv('darknet-packets-pseudoanonymous.csv', nrows=100000)

#  Pre-processing

data = data.loc[(data.src_ip != '-') & (data.src_port != '-') & (data.dst_port != '-') & (data.dst_ip != '-')]
data = data.drop(columns=['ethtype'])
imp = SimpleImputer(missing_values='-', strategy='most_frequent')

data = pd.DataFrame(imp.fit_transform(data), columns=data.columns)
data['AttID'] = data['src_ip'].apply(str).str.cat([data['src_port'].apply(str)], sep='/')


#  Identificazione attaccanti

listaAttID = data.AttID.unique()
data['ID'] = data['ts']
for element in listaAttID:
    timestamp_prossimo = 1
    i = 0
    x = i + 1
    while timestamp_prossimo is not None:
        try:
            timestamp = data['ts'].loc[(data['AttID'] == element)].iloc[i]
        except IndexError:
            break
        try:
            timestamp_prossimo = data['ts'].loc[data['AttID'] == element].iloc[x]
        except IndexError:
            break
        if (timestamp_prossimo - timestamp) < 1800:
            data['ID'].loc[(data['AttID'] == element) & (data['ts'] == timestamp_prossimo)] = timestamp
            x = x + 1
        else:
            i = x
            x = i + 1

#   Feature Creation

listaID = data.ID.unique()
data['1_port'] = data['dst_port']
data['port_list'] = data['dst_port']
data['port_list_no'] = data['dst_port']
for attacker in listaID:
    somma = 0
    concat = ''
    porta = data['dst_port'].loc[(data['ID'] == attacker)].iloc[0]
    data['port_list'].loc[(data['ID'] == attacker)] = porta
    lista = data['dst_port'].loc[(data['ID'] == attacker)].unique()
    for port in lista:
        concat = concat + port
        somma = int(port) + somma
    data['1_port'].loc[(data['ID'] == attacker)] = porta
    data['port_list'].loc[(data['ID'] == attacker)] = concat
    data['port_list_no'].loc[(data['ID'] == attacker)] = somma


encoder = LabelEncoder()
encoder.fit(data['port_list'])
data['port_list'] = encoder.transform(data['port_list'])
encoder = LabelEncoder()
encoder.fit(data['port_list_no'])
data['port_list_no'] = encoder.transform(data['port_list_no'])

#  Feature scaling

data = data.drop(columns=['AttID', 'ts', 'ID'])
standard_scaler = StandardScaler().fit(data[numeric_cols])

data1 = pd.DataFrame(StandardScaler().fit_transform(data[numeric_cols]),
                     columns=numeric_cols)
X = pd.concat([data1, data[set(data.columns).difference(numeric_cols)]], axis=1)
X_num = X.to_numpy()

#  clustering

dbscan = DBSCAN(eps=2, min_samples=22).fit(X_num)
kmean = KMeans(n_clusters=5).fit(X_num)
agglo = AgglomerativeClustering(n_clusters=5, linkage='average').fit(X_num)
agglo_labels = agglo.labels_
dbscan_labels = dbscan.labels_
kmean_labels = kmean.labels_
#X['MO'] = pd.Series(dbscan_labels, index=X.index)
X['MO'] = pd.Series(kmean_labels, index=X.index)
#data['MO'] = pd.Series(kmean_labels, index=data.index)

print(np.unique(dbscan_labels))
print(metrics.silhouette_score(X, dbscan_labels, metric='euclidean'))
print(metrics.calinski_harabasz_score(X, dbscan_labels))
print(metrics.silhouette_score(X, agglo_labels, metric='euclidean'))
print(metrics.calinski_harabasz_score(X, agglo_labels))
print(metrics.silhouette_score(X, kmean_labels, metric='euclidean'))
print(metrics.calinski_harabasz_score(X, kmean_labels))


#  Early classification


early = X[['1_port', 'MO']]
early = early[early.MO != -1]
e_X = early['1_port'].to_numpy()
e_Y = early['MO'].to_numpy()
xTrain, xTest, yTrain, yTest = train_test_split(e_X, e_Y, train_size=0.66, random_state=42)
n_units = np.count_nonzero(np.unique(kmean_labels)



model = Sequential()
model.add(Dense(5, activation='relu', input_shape=[1, ]))
model.add(Dense(5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x=xTrain, y=yTrain, validation_split=0.1, batch_size=16, epochs=5,
                    shuffle=True, verbose=2, callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3)])

print(model.summary())
pred_train = model.predict(xTrain)
scores = model.evaluate(xTrain, yTrain, verbose=0)
print('\nAccuracy of Model on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

pred_test = model.predict(xTest)
scores2 = model.evaluate(xTest, yTest, verbose=0)
print('\nAccuracy of Model on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

rounded_pred = np.argmax(pred_train, axis=1)
print("\nModel Classification Report:")
print(classification_report(yTrain, rounded_pred, digits=4))




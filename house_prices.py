import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout,Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt

print(f"TensorFlow version {tf.__version__}")
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 110)


data = pd.read_csv("C:/Users/emoradia/OneDrive - Capgemini/Desktop/Home/git/kaggle/house_prices//train.csv")
submit_data = pd.read_csv("C:/Users/emoradia/OneDrive - Capgemini/Desktop/Home/git/kaggle/house_prices/test.csv")

# make a list of the column names for categorical and numerical columns
num_columns = []
cat_columns = []

for column in data.keys():
    if data[column].dtype == object:
        cat_columns.append(column)
    else:
        num_columns.append(column)

# remove Id variable
num_columns.remove("Id")
        
print(f"Number of numeric columns: {len(num_columns)}")
print(f"Number of categorical columns: {len(cat_columns)}")
print("The target numeric variable is SalePrice")

# use sklearn model selection to split the data
train_data, test_data = train_test_split(data,test_size=0.2)

print(f"Test Samples: {len(test_data)}")
print(f"Train Samples: {len(train_data)}")

data.head()
data.describe(include="all")
data.keys().sort_values()
data.isna().sum()

# replace missing categorical values with "Missing"
for column in cat_columns:
    train_data[column].fillna(value="Missing",inplace=True)
    test_data[column].fillna(value="Missing",inplace=True)

# replace missing numerical values with the mean
for column in num_columns:
    train_data[column].fillna(train_data[column].mean(), inplace = True)
    test_data[column].fillna(test_data[column].mean(), inplace = True)

# split the train data to train and validation
train_data, val_data = train_test_split(train_data, test_size=0.2)

# scale the numeric values by standardizing
scaler = StandardScaler()
train_data[num_columns] = scaler.fit_transform(train_data[num_columns])
val_data[num_columns] = scaler.transform(val_data[num_columns])
test_data[num_columns] = scaler.transform(test_data[num_columns])

# transform pandas dataframe to tensors, shuffle and batch
def df_to_dataset(dataframe, shuffle=True,batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('SalePrice')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe),labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 32
train_ds = df_to_dataset(train_data, batch_size=batch_size)
val_ds = df_to_dataset(val_data, shuffle=False, batch_size=batch_size)
test_data['SalePrice'] = 0
test_ds = df_to_dataset(test_data, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of alleys:', feature_batch['Alley'])
    print('A batch of targets:', label_batch )

feature_columns = []

# remove SalePrice as this is the response variable
num_columns.remove('SalePrice')

# define the feature layer
for column in num_columns:
    column = tf.feature_column.numeric_column(column)
    feature_columns.append(column)

for column in cat_columns:
    column = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(column,list(train_data[column].unique())))
    feature_columns.append(column)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

def get_model():
    model = Sequential([
        feature_layer,
        Dense(64, activation="relu",name="Dense_2"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(128, activation="relu",name="Dense_3"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(256, activation="relu",name="Dense_4"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(128, activation="relu",name="Dense_5"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(64, activation="relu",name="Dense_6"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1)      
    ])
    return model
    
# print the model summary
model = get_model()

model.compile(loss='mse',
                optimizer=tf.keras.optimizers.Adam(.0001),
                metrics=['mse'])

# Create Tensorflow checkpoint objects
def get_checkpoint_every_epoch():
    checkpoint_path = 'model_checkpoints/checkpoint'
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,frequency='epoch',save_weights_only=True,
    verbose=1)
    return checkpoint

def get_early_stopping():
    early_stopping_val = EarlyStopping(monitor="loss",patience=3,verbose=1)
    return early_stopping_val

checkpoint_every_epoch = get_checkpoint_every_epoch()
early_stopping = get_early_stopping()
callbacks = [checkpoint_every_epoch, early_stopping]

history = model.fit(train_ds,validation_data=val_ds,epochs=50,callbacks=callbacks,verbose=1)

def get_test_accuracy(model, test_ds):
    test_loss, test_mse = model.evaluate(test_ds, verbose=0)
    print(f'mse: {test_mse}')
    print(f'loss: {test_loss}')

get_test_accuracy(model, test_ds)

%matplotlib inline
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

%matplotlib inline
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('MSE vs. epochs')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()









from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D, LeakyReLU, Flatten, AveragePooling2D, Reshape, Input
from tensorflow.keras.layers import Conv1D, AveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import *
from tensorflow.keras.optimizers import *
from tensorflow import keras
import time
import matplotlib.pyplot as plt

num_rows, num_columns, num_channels = 40, 1292, 1
CLASS_COUNT=23
#하이퍼파라미터
PATIENCE=300
SET = {
    'LeNet':{'epoch':300, 'batch_size':512},
    'VGG':{'epoch':300, 'batch_size':4},
    'ResNet':{'epoch':300, 'batch_size':4},
    'Logistic':{'epoch':300, 'batch_size':512},
    'SVM':{'epoch':300, 'batch_size':512}
}

# 로스, 정확도 그래프 출력
def plot_loss_accuracy(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.figure(1)
        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure(2)
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()


# 모델 셋팅
def setting(model, data_in, BATCH_SIZE, EPOCHS):
        checkpoint=ModelCheckpoint(filepath="{}/Weight_best.hdf5".format('./Saved Model/'), monitor="val_loss",
                                verbose=1, save_best_only=True)

        ealrystopping=EarlyStopping(monitor="val_loss", patience=PATIENCE)

        model.summary()

        start = time.time()
        history = model.fit(data_in.get_x_train(), data_in.get_y_train(), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(data_in.get_x_val(), data_in.get_y_val()),
                        callbacks=[checkpoint, ealrystopping], verbose=1)
        train_time = time.time() - start
        
        # plot_loss_accuracy(history)
        
        return train_time


# 모델 구조 정의
def VGG(data_in):
        EPOCHS = SET['VGG']['epoch']
        BATCH_SIZE = SET['VGG']['batch_size']
        
        model = Sequential([
                VGG16(weights=None, input_shape=(num_rows, num_columns, num_channels), include_top=False),
                Flatten(),
                # Dense(1024, activation='relu'),
                # Dense(512, activation='relu'),
                # Dense(CLASS_COUNT, activation='softmax')])
                Dense(4096, activation='relu'),
                Dense(2048, activation='relu'),
                Dense(1024, activation='relu'),
                Dense(CLASS_COUNT, activation='softmax')])
        
        model.compile(loss="categorical_crossentropy",
                metrics=["accuracy"], optimizer=SGD(learning_rate=0.0001))

        return setting(model, data_in, BATCH_SIZE, EPOCHS)


def LeNet(data_in):
        EPOCHS = SET['LeNet']['epoch']
        BATCH_SIZE = SET['LeNet']['batch_size']
        
        model = Sequential([
                Input(shape=(num_rows, num_columns, num_channels)),
                Conv2D(6, kernel_size=(5,5), strides=(1,1), padding="same", activation='relu'),
                AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
                Conv2D(16, kernel_size=(5,5), strides=(1,1), padding="same", activation='relu'),
                AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
                Flatten(),
                Dense(120, activation='relu'),
                Dense(84, activation='relu'),
                Dense(CLASS_COUNT, activation='softmax')])
        
        model.compile(loss="categorical_crossentropy",
                metrics=["accuracy"], optimizer=Adam())

        return setting(model, data_in, BATCH_SIZE, EPOCHS)


def ResNet(data_in):
        EPOCHS = SET['ResNet']['epoch']
        BATCH_SIZE = SET['ResNet']['batch_size']
        
        model = Sequential([
                ResNet50(weights=None, input_shape=(num_rows, num_columns, num_channels), include_top=False),
                Flatten(),
                Dense(1024, activation='relu'),
                Dense(512, activation='relu'),
                Dense(CLASS_COUNT, activation='softmax')])
        
        model.compile(loss="categorical_crossentropy",
                metrics=["accuracy"], optimizer=Adam())

        return setting(model, data_in, BATCH_SIZE, EPOCHS)


def SVM(data_in):
        EPOCHS = SET['SVM']['epoch']
        BATCH_SIZE = SET['SVM']['batch_size']
        
        model = Sequential([
                Input(shape=(num_rows * num_columns)),
                Dense(CLASS_COUNT, activation=None)])
        
        model.compile(loss="CategoricalHinge",
                metrics=["accuracy"], optimizer=Adam(learning_rate=5e-4))

        return setting(model, data_in, BATCH_SIZE, EPOCHS)


def Logistic(data_in):
        EPOCHS = SET['Logistic']['epoch']
        BATCH_SIZE = SET['Logistic']['batch_size']
        
        model = Sequential([
                # Input(shape=(num_rows * num_columns)),
                Dense(CLASS_COUNT, activation='softmax', input_shape=(None, num_rows * num_columns))
                ])
        
        model.compile(loss="categorical_crossentropy",
                metrics=["accuracy"], optimizer=Adam(learning_rate=1e-3))

        return setting(model, data_in, BATCH_SIZE, EPOCHS)
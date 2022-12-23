from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import mae
from tensorflow.keras.optimizers import Adam
from data import prepare_data
import matplotlib.pyplot as plt

''' 
Build Model with Keras Sequentia API
'''
def build_model():
    model = Sequential(
        [
            Dense(32, input_shape=(79,), activation='relu'),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='relu'),
        ]
    )

    return model

if __name__ == '__main__':
    # define size of validation set during training
    test_split = 0.0
    # prepare training and validation data and labels for training
    X_train, X_validate, y_train, y_validate, _ = prepare_data(['train.csv', 'test.csv'], test_split=test_split)

    print(f'X Train: {X_train}')
    print(f'y Train: {y_train}')
    print(f'X Validate: {X_validate}')
    print(f'y Validate: {y_validate}')

    # build DNN Model
    model = build_model()

    # compile model with mean average error as loss function and setting Adam as the optimizer
    model.compile(loss=mae, optimizer=Adam(learning_rate=1e-5), metrics=['mse'])
    
    # fit the model
    history = model.fit(X_train, y_train, batch_size=32, epochs=2000, validation_data=(X_validate, y_validate))

    # save weights of model
    model.save_weights('./checkpoints/my_checkpoint6')

    # plot losses and validation losses
    plt.plot(history.history['loss'])
    if test_split:
        plt.plot(history.history['val_loss'])

    plt.show()
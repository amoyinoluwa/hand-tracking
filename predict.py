import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
import numpy as np
from data_processing import DataProcessor
from sklearn.preprocessing import LabelEncoder

class HandMovementPredictor:
    def __init__(self) -> None:
        self.num_classes = 3
        data_processor = DataProcessor()
        encoder = LabelEncoder()
        self.X, self.y = data_processor.process_data()
        y_one_hot = encoder.fit_transform(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, y_one_hot, test_size=0.2, random_state=42)

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.X.shape[1], self.X.shape[2], 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=tf.keras.losses.sparse_categorical_crossentropy,
                    metrics=['accuracy'])
        return model

    def train_model(self, save_model_checkpoint=True):
        model = self.create_model()
        if not save_model_checkpoint:
            model.fit(self.X_train, self.y_train, epochs=50, validation_split=0.15)
            return
        model_checkpoint = 'ml_model.ckpt'
        # _callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_checkpoint,
        #                                                 save_weights_only=True,
        #                                                 verbose=1)
        model.fit(self.X_train, self.y_train, epochs=50, validation_split=0.15)
        model.save('pretrained')

predictor = HandMovementPredictor()
# model = predictor.train_model()
        
new_model = tf.keras.models.load_model('pretrained')
preds = new_model.predict(predictor.X_test)
print(np.argmax(preds, axis=1))

#sample test
# handPredictor = HandMovementPredictor()
# handPredictor.train_model()
# predictions = handPredictor.predict(X_test)
# predicted_classes = np.argmax(predictions, axis=1)

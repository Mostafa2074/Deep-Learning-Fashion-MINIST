import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils import Preprocessing

def build_model():
    model = Sequential([
        Dense(512, input_shape=(784,)),
        LeakyReLU(0.1),
        BatchNormalization(),
        Dropout(0.25),

        Dense(512),
        LeakyReLU(0.1),
        BatchNormalization(),
        Dropout(0.25),

        Dense(256),
        LeakyReLU(0.1),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128),
        LeakyReLU(0.1),
        BatchNormalization(),
        Dropout(0.25),

        Dense(10, activation='softmax')
    ])
    return model

def train(x_train, y_train, x_test, y_test):
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=40,
        batch_size=128,
        callbacks=[reduce_lr],
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
    print(f"✅ Test Loss: {test_loss:.4f}")

    return model

if __name__ == "__main__":
    p = Preprocessing()
    p.load_and_prepare()

    model = train(p.x_train, p.y_train, p.x_test, p.y_test)
    p.save_model(model)

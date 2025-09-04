from tensorflow.keras import layers, models

def build_model(input_shape=(150, 150, 3)):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # Flatten
    model.add(layers.Flatten())

    # Dense layers with Dropout
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))   # 🔥 Dropout added

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

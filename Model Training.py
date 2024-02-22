####### IMPORTS #############
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix, classification_report
from plot_cm import plot_confusion_matrix

##### Loading saved csv ##############
df = pd.read_csv(r"D:\AI Assistant\Wake word detection\final_audio_data.csv")

# Convert string representation of arrays back to arrays
df['feature'] = df['feature'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# Extract features and labels
X = np.vstack(df['feature'])  # Stack feature arrays vertically to form X
y = df['class_label'].values

# Ensure the shape of X
print("Shape of X before reshaping:", X.shape)

# Reshape X to match the desired shape (number_of_samples, number_of_features)
X = X.reshape(X.shape[0], -1)  # Reshape X to have 40 features per sample

# Ensure the shape of X after reshaping
print("Shape of X after reshaping:", X.shape)

# Convert integer labels to one-hot encoded vectors
num_classes = 2  # Assuming you have 2 classes
y = to_categorical(y, num_classes)

####### train test split ############
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

##### Training ############

model = Sequential([
    Dense(256, input_shape=X_train[0].shape),
    Activation('relu'),
    Dropout(0.5),
    Dense(256),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

print(model.summary())

model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)

print("Model Score: \n")
history = model.fit(X_train, y_train, epochs=1000)
model.save("WWD.h5")
score = model.evaluate(X_test, y_test)
print(score)

#### Evaluating our model ###########
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=["Does not have Wake Word", "Has Wake Word"])

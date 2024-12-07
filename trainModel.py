from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential

data_dir = "Animals/"

datagen = ImageDataGenerator(
    rescale=1./255,      
    validation_split=0.2 
)


train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',            
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu',padding="same", input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3),padding="same", activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3),padding="same", activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3),padding="same", activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax') 
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',  # Since this is multi-class classification
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10 
)

test_loss, test_accuracy = model.evaluate(test_generator)
train_loss, train_accuracy = model.evaluate(train_generator)
print("Test accuracy:" ,test_accuracy)
print("Train accuracy:" ,train_accuracy)
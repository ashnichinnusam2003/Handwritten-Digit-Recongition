import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

#PREPROCESSING PART

mnist=tf.keras.datasets.mnist # loading the dataset mnist which contain the handwritten digit
(x_train,y_train),(x_test,y_test)=mnist.load_data() # divide the datset into training and testing data.


#CNN PART

model = tf.keras.models.Sequential() # created the model
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # added a flatten layer
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy']) #compile the model.

##TRAIN THE MODEL

model.fit(x_train,y_train,epochs=3)
model.save('Handwritten.model')

#_____________________________________________________
# can use the code below as instead of the entire code..
model = tf.keras.models.load_model('Handwritten.model')
loss,accuracy = model.evaluate(x_test,y_test)
print(loss)
print(accuracy)
#_____________________________________________________


image_number=1
while os.path.isfile(f"digits/{image_number}.png"):
    try:
        image=cv2.imread(f"digits/{image_number}.png")[:,:,0]
        image=np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_number +=1
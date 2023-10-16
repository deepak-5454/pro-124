import cv2
import numpy as np
import tensorflow as tf

# Load a pre-trained hand gesture recognition model (e.g., MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (224, 224))
    frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)

    # Make a prediction
    prediction = model.predict(frame)
    gesture_label = np.argmax(prediction)

    # Display the recognized gesture label on the frame
    cv2.putText(frame, "Gesture: " + str(gesture_label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

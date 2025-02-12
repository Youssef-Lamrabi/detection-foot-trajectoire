import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def test_model():
    # Load the model
    model = tf.keras.models.load_model('mnist_model.h5')
    
    # Load MNIST test data
    (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Preprocess the test data
    X_test = X_test / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    accuracy = np.sum(y_pred_classes == y_test) / len(y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Test some random samples
    test_samples = np.random.randint(0, len(X_test), 5)
    for idx in test_samples:
        prediction = model.predict(X_test[idx:idx+1])
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100
        
        print(f"\nSample {idx}:")
        print(f"True digit: {y_test[idx]}")
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Display the image
        plt.figure(figsize=(3, 3))
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Predicted: {predicted_digit} (True: {y_test[idx]})')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    test_model()
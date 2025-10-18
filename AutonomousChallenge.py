
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def generate_traffic_heatmaps(samples=500, size=64):
    data = []
    for _ in range(samples):
        base = np.zeros((size, size))
        # simulate random "roads" or dense regions
        for _ in range(np.random.randint(3, 8)):
            x, y = np.random.randint(0, size, 2)
            base += np.exp(-((np.arange(size) - x)[:, None]**2 + (np.arange(size) - y)[None, :]**2) / (2*(np.random.randint(2, 6)**2)))
        base /= np.max(base)
        data.append(base)
    return np.array(data)[..., None]
traffic_data = generate_traffic_heatmaps(samples=800, size=64)
print(f"Traffic dataset shape: {traffic_data.shape}")

input_shape = (64, 64, 1)
autoencoder = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), padding='same'),
    layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), padding='same'),
    layers.Conv2D(8, (3,3), activation='relu', padding='same'),

    layers.UpSampling2D((2,2)),
    layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    layers.UpSampling2D((2,2)),
    layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

autoencoder.fit(traffic_data, traffic_data, epochs=20, batch_size=16, validation_split=0.1)

test_sample = traffic_data[np.random.randint(0, len(traffic_data))]
reconstructed = autoencoder.predict(test_sample[None, ...])[0].squeeze()
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(test_sample.squeeze(), cmap='inferno')
plt.title("Original Traffic Heatmap")
plt.axis('off')
plt.show()

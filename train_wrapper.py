import sys
sys.path.insert(0, '.')

from models.calibration_net.train import train_model

# Train CNN model
print("Training CNN architecture...")
train_model(architecture='cnn', epochs=30, batch_size=32, learning_rate=0.001)

import sys
sys.path.insert(0, '.')

from models.calibration_net.train import train_model

# Train ResNet model
print("Training ResNet architecture...")
train_model(architecture='resnet', epochs=30, batch_size=32, learning_rate=0.001)
print("\nResNet training complete!")

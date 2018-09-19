# needs model loss, accuracy, data, they extend compressible models
from models.mnist import Lenet5
from models.cifar10 import VGG
import numpy as np

# Training LeNet-5 on MNIST
model = Lenet5(bpb=10)

model.train(200000, False)
model.train(200000, True)
# retrain,
print(model.compress(100))

# Training VGG-16 on CIFAT-10
model = VGG(bits=10)

model.train(100000, False)
model.train(150000, True)
# retrain,
print(model.compress(1))



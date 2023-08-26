from Engine import Value, Neuron, MLP, Visualizer
from MLUtils import printPurple

# initialize a model 
model = MLP(3, [4, 4, 1]) # 2-layer neural network with 4 neuron in each hidden layer

# Dataset: for each xs as an input the output should be the corresponding ys. So 
# e.g. if input is [0.5, 1.0, 1.0] --> -1.0
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [0.25, 1.0, 1.0],
    [0.25, 0.75, 1.0],
]

ys = [1.0, -1.0, -1.0, 1.0, 0.5, -0.5]

# Specify Epochs and Learning Rate
epochs = 10
iterPerEpoch = 100
lr = 0.05

def train():
    initialLoss = sum((yOut + (-yGroundTruth)) ** 2 for yGroundTruth, yOut in zip(ys, [model(x) for x in xs]))
    for i in range(epochs):
        for k in range(iterPerEpoch):
            ypred = [model(x) for x in xs]
            loss = sum((yOut + (-yGroundTruth)) ** 2 for yGroundTruth, yOut in zip(ys, ypred)) # Forward Propagation
            
            loss.label = "Loss"
            
            for param in model.parameters():
                param.grad = 0.0 # Zero the gradient
                
            loss.backProp() # Back Propagation
            
            for param in model.parameters():
                param.value +=  param.grad * -lr # Update the parameters by a small factor
                
        print(f'Epoch {i+1} Loss: {loss.value:.6f}')
    printPurple(f"\nSummary of Training: \n---------------------")
    printPurple(f"Initial Loss: {initialLoss.value:.6f} \nFinal Loss: {loss.value:.6f} \nTotal Improvement: {(initialLoss.value - loss.value):.6f}")
train()
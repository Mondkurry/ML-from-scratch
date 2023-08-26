from Engine import Value, Neuron, MLP, Visualizer
from MLUtils import loadingAnimation
from tqdm import tqdm

# initialize a model 
model = MLP(3, [4, 4, 1]) # 2-layer neural network with 4 neuron in each hidden layer

# Dataset: for each xs as an input the output should be the corresponding ys. So 
# e.g. if input is [0.5, 1.0, 1.0] --> -1.0
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]

# Specify Epochs and Learning Rate
epochs = 1000
lr = 0.05

def train():
    for k in tqdm(range(epochs), desc=f"Training MLP", ascii=False, ncols=75):
        ypred = [model(x) for x in xs]
        loss = sum((yOut + (-yGroundTruth)) ** 2 for yGroundTruth, yOut in zip(ys, ypred)) # Forward Propagation
        
        loss.label = "Loss"
        
        for param in model.parameters():
            param.grad = 0.0 # Zero the gradient
            
        loss.backProp() # Back Propagation
        
        for param in model.parameters():
            param.value +=  param.grad * -lr # Update the parameters by a small factor
            
    print (f'Epoch {k+1} Loss: {loss.value:.6f}')
        
train()
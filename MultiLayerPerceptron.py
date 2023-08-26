from Engine import Value, Neuron, MLP, Visualizer
from MLUtils import printPurple
import random

def train(model, xs, ys, epochs, iterPerEpoch, lr):
    initialLoss = sum((yOut + (-yGroundTruth)) ** 2 for yGroundTruth, yOut in zip(ys, [model(x) for x in xs]))
    print("\n\033[1;33mTraining: \n------------------------\033[0m")
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
                
        print(f'Epoch {i+1} Loss: \033[1;33m{loss.value:.6f}\033[0m')
    print(f"\n\033[1;33mSummary of Training: \n------------------------\033[0m")
    print(f"Initial Loss: \t{initialLoss.value:.6f}\t \nFinal Loss: \t{loss.value:.6f}\t \nImprovement: \t{(initialLoss.value - loss.value):.6f}\t")
    


def demo(model, xs, ys):
    trydemo = input("\nWould you like to try the model? (y/n): ")
    
    while trydemo == 'y':
        print(f"\n\033[1;33mDemo: \n------------------------\033[0m")
        print(f"dataSet: \n")
        for index, value in enumerate(xs):
            print(index, ":\t", value, "\t----->\t", ys[index])
        
        index = input("\nEnter an index of the dataset (0-5): ")
        
        if int(index) > 5:
            index = random.randint(0, len(xs) - 1)
            print(f"That index was out of range, so I chose {index} for you!")
        
        roundedto25 = round(model(xs[int(index)]).value*4, 0) / 4
        groundTruth = ys[int(index)]
        print(f"\nPrediction:\t{roundedto25} \nGround Truth:\t{groundTruth}")
                
        if float(roundedto25) == float(groundTruth):
            print("\033[1;32mThe model got it Correct!\033[0m")
        else:
            print("\033[1;31mThe model got it Wrong!\033[0m")
            
        trydemo = input("\nWould you like to try the model again? (y/n): ")
            
def main():

    model = MLP(3, [4, 5, 1]) # 2-layer MLP network with 4 neurons in each hidden layer

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

    # Specify Epochs, iterations per epoch and and Learning Rate
    train(model, xs, ys, 10, 100, 0.05)
    demo(model, xs, ys,)

if __name__ == "__main__":
    main()
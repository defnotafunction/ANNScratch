import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

RNG = np.random.default_rng()

def ReLU(number: int):
        """Returns max value between number and 0"""
        return max(0, number)
    

class MSE:
    def __new__(self, y_pred_values: np.ndarray, y_true_values: np.ndarray):
        """Returns the mean of the squares of the differences between the predicted and true values"""
        if not isinstance(y_pred_values, np.ndarray) or not isinstance(y_true_values, np.ndarray):
            y_pred_values = np.array(y_pred_values)
            y_true_values = np.array(y_true_values)
        
        differences = y_pred_values - y_true_values
        squared_differences = np.sum(differences ** 2)
        mean_squared_difference = squared_differences / len(differences)
        
        return mean_squared_difference 
    



class Perceptron:
    def __init__(self, input_neurons: int, activation_function: callable = None):
        """A basic network structure"""
        self.input_neurons = input_neurons
        self.activation_function = activation_function
        self.weights = RNG.random(self.input_neurons)
        self.bias = RNG.random()
    
    def predict(self, examples: np.ndarray, weights: np.ndarray = None, bias: np.ndarray = None):
        predictions = []
        weights_to_use = self.weights if weights is None else weights
        bias_to_use = self.bias if bias is None else bias

        for X in examples:
            if len(X) != len(weights_to_use):
                raise ValueError('Lengths of examples and weights were not equal')

            weighted_sum = X * weights_to_use
            weighted_sum = np.sum(weighted_sum) + bias_to_use
            
            if self.activation_function:
                weighted_sum = self.activation_function(weighted_sum)
            
            predictions.append(weighted_sum)
        
        return predictions
    
    def monkey_step(self, X: list[np.ndarray], y_true_values: np.ndarray, loss_function: callable, learning_rate: float):
        # Optimization but less complex math
        # Steps by getting the loss from using the new weight / bias and subtracting it by the old loss 

        for idx, w in enumerate(self.weights):
            baseline_prediction = self.predict(X)
            baseline_loss = loss_function(baseline_prediction, y_true_values)

            new_weights = self.weights.copy()
            new_weights[idx] = w + 0.001
            new_prediction = self.predict(X, weights=new_weights)
            new_loss = loss_function(new_prediction, y_true_values)

            direction = (new_loss - baseline_loss) / 0.001
            self.weights[idx] = w - direction * learning_rate
        
        # Optimizing Bias

        baseline_prediction = self.predict(X)
        baseline_loss = loss_function(baseline_prediction, y_true_values)

        new_bias = self.bias + 0.001
        new_prediction = self.predict(X, bias=new_bias)
        new_loss = loss_function(new_prediction, y_true_values)

        direction = (new_loss - baseline_loss) / 0.001
        self.bias -= direction * learning_rate

class ANN:
    def __init__(self, neuron_sets: list[tuple[int, int]]):
        self.neuron_sets = neuron_sets
        self.layers = self.get_layers()
    
    def show_structure(self):
        for layer_idx, layer in enumerate(self.layers):
            print(f'=========== LAYER {layer_idx + 1} ===========')
            for p in layer:
                print(f' - Neuron: (Weights: {p.weights}, Bias: {p.bias}, AF: {p.activation_function})')

    def get_layers(self):
        layers = [[] for neuron_set in self.neuron_sets if isinstance(neuron_set, tuple)]  # Filters out activation functions 
        layer_idx = 0
        
        for set_idx, neuron_set in enumerate(self.neuron_sets):
            if isinstance(neuron_set, tuple):
                activation_function = None

                # Check for activation function
                try:
                    if isinstance(self.neuron_sets[set_idx+1], function): 
                        activation_function = self.neuron_sets[set_idx+1]
                except:
                    pass

                for i in range(neuron_set[1]):
                    perceptron = Perceptron(input_neurons=neuron_set[0], activation_function=activation_function)
                    layers[layer_idx].append(perceptron)
                
                layer_idx += 1
    
        return layers
    
    def forward(self, X):
        for layer_idx, layer in enumerate(self.layers):
            current_neuron_numbers = []

            for p in layer:
                if layer_idx == 0:
                    current_neuron_numbers.append(p.predict(X)[0])
                else:
                    current_neuron_numbers.append(p.predict([neuron_numbers])[0])

            neuron_numbers = np.array(current_neuron_numbers[:])

        output = neuron_numbers
        return output




ann = ANN([
    (5, 16),
    ReLU,
    (16, 8),
    ReLU,
    (8, 2)
    ])
print('========== ANN ============')
print(ann.layers)
print(ann.forward([[6, 4, 5, 3, 6]]))
ann.show_structure()

            
            


# TESTING

X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
neuron = Perceptron(1)



print('=========== BEFORE TRAINING ===============')
og_predictions = neuron.predict(X_test)
print(neuron.weights)
print(neuron.bias)
print(MSE(og_predictions, y_test))

for i in range(1000):
    neuron.monkey_step(X_train, y_train, MSE, 0.01)

print('=========== AFTER TRAINING ===============')
new_predictions = neuron.predict(X_test)
print(neuron.weights)
print(neuron.bias)
print(MSE(new_predictions, y_test))

plt.suptitle('Perceptron Training')
plt.subplot(1, 2, 1)
plt.title('Before Training')
plt.scatter(X_test, y_test, c='Red', alpha=0.2)
plt.plot(X_test, og_predictions, color='Green')

plt.subplot(1, 2, 2)
plt.title('After Training')
plt.scatter(X_test, y_test, c='Red', alpha=0.2)
plt.plot(X_test, new_predictions, color='Blue')
print(X_test)
print(og_predictions)
print(new_predictions)
plt.show()

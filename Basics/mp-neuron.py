"""The NP Neuron (more commonly known as the McCulloch-Pitts (M-P) Neuron) 
was the very first mathematical model of a biological neuron, proposed back in 1943"""

class MPNeuron:
    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, inputs):
        # Calculate the sum of inputs (assuming weights are 1)
        total_sum = sum(inputs)
        
        # Apply the step function based on the threshold
        return 1 if total_sum >= self.threshold else 0



# Test Data: 2-bit combinations (0,0), (0,1), (1,0), (1,1)
input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]

and_gate = MPNeuron(threshold=2)

print("AND Gate Truth Table:")
for x in input_data:
    print(f"Input: {x} -> Output: {and_gate.predict(x)}")

or_gate = MPNeuron(threshold=1)

print("\nOR Gate Truth Table:")
for x in input_data:
    print(f"Input: {x} -> Output: {or_gate.predict(x)}")



class MPNeuronInhibitory:
    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, excitatory_inputs, inhibitory_inputs):
        # If any inhibitory input is 1, the output is always 0
        if any(inhibitory_inputs):
            return 0
        
        # Otherwise, check the threshold for excitatory inputs
        return 1 if sum(excitatory_inputs) >= self.threshold else 0

# Example: A AND NOT B
# Threshold = 1 (for input A), but if B is 1, it inhibits the output.
neuron = MPNeuronInhibitory(threshold=1)
print(f"\nExcitatory(1), Inhibitory(1) -> Output: {neuron.predict([1], [1])}") # Returns 0
print(f"Excitatory(1), Inhibitory(0) -> Output: {neuron.predict([1], [0])}") # Returns 1
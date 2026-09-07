def mp_neuron_nand(x1, x2):
    # Define weights and threshold
    w1, w2 = -2, -2
    threshold = 1
    
    # Calculate net input
    net_input = (x1 * w1) + (x2 * w2)
    
    # Threshold activation function
    return 1 if net_input >= threshold else 0

# Test all combinations
inputs = [(0,0), (0,1), (1,0), (1,1)]
for x1, x2 in inputs:
    print(f"Inputs: ({x1}, {x2}) -> FASLE Output: {mp_neuron_nand(x1, x2)}")

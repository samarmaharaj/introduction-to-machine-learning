"""Generate two-input Boolean functions with a classical MP neuron."""


class MPNeuron:
    """A classical MP neuron with excitatory and inhibitory connections."""

    def __init__(self, excitatory, inhibitory, threshold):
        self.excitatory = tuple(excitatory)
        self.inhibitory = tuple(inhibitory)
        self.threshold = threshold

    def predict(self, inputs):
        if any(inputs[index] for index in self.inhibitory):
            return 0
        return int(sum(inputs[index] for index in self.excitatory) >= self.threshold)


INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
FUNCTIONS = {
    "FALSE":  "0000", "AND":   "0001", "A AND NOT B": "0010", "A": "0011",
    "NOT A AND B": "0100", "B": "0101", "XOR":   "0110", "OR": "0111",
    "NOR":    "1000", "XNOR":  "1001", "NOT B": "1010", "A OR NOT B": "1011",
    "NOT A":  "1100", "NOT A OR B": "1101", "NAND": "1110", "TRUE": "1111",
}


def find_neuron(expected):
    # Each input is disconnected, excitatory, or inhibitory; no weights.
    for connection_a in (None, "excitatory", "inhibitory"):
        for connection_b in (None, "excitatory", "inhibitory"):
            connections = (connection_a, connection_b)
            excitatory = tuple(i for i, c in enumerate(connections)
                               if c == "excitatory")
            inhibitory = tuple(i for i, c in enumerate(connections)
                               if c == "inhibitory")
            for threshold in range(0, 3):
                neuron = MPNeuron(excitatory, inhibitory, threshold)
                if "".join(str(neuron.predict(x)) for x in INPUTS) == expected:
                    return neuron
    return None


solutions = {name: find_neuron(outputs) for name, outputs in FUNCTIONS.items()}
headers = ["A", "B"] + list(FUNCTIONS)
print("\t".join(headers))
for a, b in INPUTS:
    print("\t".join([str(a), str(b)] + [str(int(outputs[a * 2 + b])) for outputs in FUNCTIONS.values()]))

print("\nMP neuron implementations:")
for name, neuron in solutions.items():
    result = (f"excitatory={neuron.excitatory}, "
              f"inhibitory={neuron.inhibitory}, threshold={neuron.threshold}"
              if neuron else "NOT POSSIBLE with one MP neuron")
    print(f"{name}: {result}")
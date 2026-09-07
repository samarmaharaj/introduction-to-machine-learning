"""Single-layer perceptron for all 16 Boolean functions of two variables."""

from itertools import product


INPUTS = list(product((0, 1), repeat=2))  # 00, 01, 10, 11


def train_perceptron(outputs, max_epochs=100):
    """Return (weights, bias), or None when the function is not separable."""
    weights = [0, 0]
    bias = 0

    for _ in range(max_epochs):
        errors = 0
        for (x1, x2), expected in zip(INPUTS, outputs):
            prediction = int(weights[0] * x1 + weights[1] * x2 + bias >= 0)
            error = expected - prediction
            if error:
                weights[0] += error * x1
                weights[1] += error * x2
                bias += error
                errors += 1
        if errors == 0:
            return weights, bias
    return None


def predict(inputs, weights, bias):
    x1, x2 = inputs
    return int(weights[0] * x1 + weights[1] * x2 + bias >= 0)


not_implementable = []

for function_number in range(16):
    # Bit 0 is the result for 00; bit 3 is the result for 11.
    outputs = [(function_number >> row) & 1 for row in range(4)]
    model = train_perceptron(outputs)

    if model is None:
        not_implementable.append(function_number)
        print(f"F{function_number:02d}: {outputs} -> NOT IMPLEMENTABLE by one perceptron")
    else:
        weights, bias = model
        predictions = [predict(row, weights, bias) for row in INPUTS]
        print(
            f"F{function_number:02d}: {predictions} "
            f"(weights={weights}, bias={bias})"
        )

print("\nNot implementable by a single perceptron:")
print(" ".join(f"F{number:02d}" for number in not_implementable))
print("These are XOR (F06) and XNOR (F09); they are not linearly separable.")
# Neural Arithmetic Logic Units

[WIP]

This is a PyTorch implementation of [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508) by *Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer and Phil Blunsom*.

<p align="center">
 <img src="./imgs/arch.png" alt="Drawing", width=60%>
</p>

## API

```python
from models import *

# single layer modules
NeuralAccumulatorCell(in_dim, out_dim)
NeuralArithmeticLogicUnitCell(in_dim, out_dim)

# stacked layers
NAC(num_layers, in_dim, hidden_dim, out_dim)
NALU(num_layers, in_dim, hidden_dim, out_dim)
```

## Experiments

To reproduce "Numerical Extrapolation Failures in Neural Networks" (Section 1.1), run:

```python
python failures.py
```

This should generate the following plot:

<p align="center">
 <img src="./imgs/extrapolation.png" alt="Drawing", width=60%>
</p>

To reproduce "Simple Function Learning Tasks" (Section 4.1), run:

```python
python function_learning.py
```
This should generate a text file called `interpolation.txt` with the following results. (Currently only supports interpolation, I'm working on the rest)

|         | Relu6    | None     | NAC      | NALU   |
|---------|----------|----------|----------|--------|
| a + b   | 4.472    | 0.132    | 0.154    | 0.157  |
| a - b   | 85.727   | 2.224    | 2.403    | 34.610 |
| a * b   | 89.257   | 4.573    | 5.382    | 1.236  |
| a / b   | 97.070   | 60.594   | 5.730    | 3.042  |
| a ^ 2   | 89.987   | 2.977    | 4.718    | 1.117  |
| sqrt(a) | 5.939    | 40.243   | 7.263    | 1.119  |

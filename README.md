# micrograd

![micrograd](micrograd.png)

A tiny scalar-valued autograd engine and neural network library, built from scratch while following [Karpathy's micrograd](https://github.com/karpathy/micrograd) lecture.

## Structure

```
micrograd/
├── engine.py   # Value: scalar autograd (backprop over a dynamic DAG)
├── nn.py       # Neuron, Layer, MLP
└── draw.py     # draw_dot (computation graph), draw_mlp (architecture), logging helpers
```

## Examples

| Dir | What it does |
|---|---|
| `neuralNetwork1/` | Single neuron computation graph via `draw_dot` |
| `neuralNetwork2/` | Neuron with tanh activation, graph visualization |
| `neuralNetwork3/` | MLP `[3 → 4 → 4 → 1]` trained on 4 samples; arch + graph SVGs, training table |
| `neuralNetwork4/` | Binary classifier on `make_moons`; decision boundary plot |


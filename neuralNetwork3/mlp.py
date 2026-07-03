from micrograd.nn import MLP
from micrograd.engine import Value
from micrograd.draw import draw_mlp, draw_dot, TrainingLog

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 0.5, 1.0], [1.0, 1.0, -1.0]]
n = MLP(3, [4, 4, 1])
ys = [1.0, -1.0, -1.0, 1.0]


log = TrainingLog(ys)  # writes training_log.txt 

for k in range(20):

    # forwad pass
    ypred = [n(x) for x in xs]
    loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)), Value(0.0))  # type: ignore

    # flush the gradient
    for p in n.parameters():
        p.grad = 0.0

    # backward pass
    loss.backward()

    # nudge all the neurons towards the minimization of the loss
    for p in n.parameters():
        p.data += -0.05 * p.grad

    log.row(k, loss, ypred)

log.close()

"""
    the below part is just for visual
"""

# visualize the trained network (just the mlp architecture )
draw_mlp(n).render("mlp_arch", format="svg", view=True)

# same net shape as mlp.py
n = MLP(3, [4, 4, 1])
x = [2.0, 3.0, -1.0]
out = n(x)          # one forward pass -> single Value
dot = draw_dot(out)
dot.render("mlp_graph", view=True)

from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )

        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot


class RunLog:
    """Print lines to console and mirror them to a text file.

    Usage:
        log = RunLog("moons_log.txt")
        log(model)                      # any object, like print
        log(f"step {k} loss {l}")
        log.close()
    """

    def __init__(self, path="run_log.txt"):
        self.file = open(path, "w")

    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        print(line)
        self.file.write(line + "\n")

    def close(self):
        self.file.close()


class TrainingLog:
    """Training table printed to console and written to a text file.

    Usage:
        log = TrainingLog(ys)          # header
        log.row(k, loss, ypred)        # one row per step
        log.close()                    # targets row + close file
    """

    def __init__(self, ys, path="training_log.txt"):
        self.ys = ys
        self.file = open(path, "w")
        self._rule = "-" * (4 + 3 + 10 + len(ys) * 11)
        self._log(f"{'step':>4} | {'loss':>10} | " + " | ".join(f"{'ypred%d' % i:>8}" for i in range(len(ys))))
        self._log(self._rule)

    def _log(self, line):
        print(line)
        self.file.write(line + "\n")

    def row(self, k, loss, ypred):
        self._log(f"{k:>4} | {loss.data:>10.6f} | " + " | ".join(f"{yp.data:>8.4f}" for yp in ypred))

    def close(self):
        self._log(self._rule)
        self._log(f"{'ygt':>4} | {'':>10} | " + " | ".join(f"{y:>8.4f}" for y in self.ys))
        self.file.close()


def draw_mlp(model):
    """Architecture diagram of an MLP: neurons as circles, layers as columns.

    Edge thickness = |weight|, blue = positive, red = negative.
    """
    nin = len(model.layers[0].neurons[0].w)
    sizes = [nin] + [len(layer.neurons) for layer in model.layers]

    g = Digraph(
        "mlp",
        graph_attr={"rankdir": "LR", "splines": "line", "nodesep": "0.3"},
    )

    # one node id per neuron, per layer
    layer_nodes = []
    for li, size in enumerate(sizes):
        ids = []
        for j in range(size):
            uid = "L%d_%d" % (li, j)
            if li == 0:
                g.node(uid, "x%d" % j, shape="circle", style="filled", fillcolor="#cde")
            elif li == len(sizes) - 1:
                g.node(uid, "out", shape="circle", style="filled", fillcolor="#dfc")
            else:
                g.node(uid, "", shape="circle", style="filled", fillcolor="#eee")
            ids.append(uid)
        layer_nodes.append(ids)

    # weighted edges with the weight value as a label
    for li, layer in enumerate(model.layers):
        for j, neuron in enumerate(layer.neurons):
            dst = layer_nodes[li + 1][j]
            for i, wi in enumerate(neuron.w):
                src = layer_nodes[li][i]
                pen = 0.3 + abs(wi.data) * 2
                color = "#c33" if wi.data < 0 else "#39c"
                g.edge(src, dst, penwidth="%.2f" % pen, color=color)

    return g

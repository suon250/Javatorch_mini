package mlp;

public class ReLU implements Op {
    public Tensor forward(Tensor x) {
        double[] out = new double[x.size];
        boolean[] mask = new boolean[x.size];

        for (int i = 0; i < x.size; i++) {
            if (x.data[i] > 0) {
                out[i] = x.data[i];
                mask[i] = true;
            }
        }

        Tensor y = new Tensor(out, x.shape, x.requiresGrad);
        y.creator = this;
        y.parents = new Tensor[]{x};
        y.ctx = mask;

        return y;
    }

    @Override
    public void backward(Tensor y) {
        Tensor x = y.parents[0];
        boolean[] mask = (boolean[]) y.ctx;

        for (int i = 0; i < x.size; i++) {
            x.grad[i] += y.grad[i] * (mask[i] ? 1.0 : 0.0);
        }
    }
}

package mlp;

public class Tensor {
    // store the value of data
    public double[] data;        // flatten in 1D
    public double[] grad;        // same shape as data

    // shape of actual data (before flattening)
    public int[] shape;          // e.g. [784] or [128, 784]
    public int size;             // data.length

    // autograd
    public boolean requiresGrad;
    public Op creator;           // the abstract layer(regardless of actual data value)
    public Tensor[] parents;     // the value of data(along with creator with datatype Op)

    // cache（created for backward）???
    public Object ctx;
    
    // basic constructor(used for the flattened 1D MNIST data 0-9)
    public Tensor(double[] data, int[] shape, boolean requiresGrad) {
        this.data = data;
        this.shape = shape;
        this.size = data.length;
        this.requiresGrad = requiresGrad;
        this.grad = requiresGrad ? new double[size] : null;
    }
    
    // clear up the gradient
    public void zeroGrad() {
    	for (int i = 0; i < grad.length; i++) {
    		grad[i] = 0;
    	}
    }
    
    // add other useful tensor !!!!
    // Tensor zeros(int[] shape)
    // Tensor randn(int[] shape)
    
    public void backward() {
        if (!requiresGrad) return;

        // 默认认为是 scalar loss
        grad[0] = 1.0;

        backwardRecursive(this);
    }
    
    private static void backwardRecursive(Tensor t) {
        if (t.creator == null) return;

        t.creator.backward(t);

        for (Tensor p : t.parents) {
            if (p.requiresGrad) {
                backwardRecursive(p);
            }
        }
    }


}


package mlp;

public interface Op {
	public void backward(Tensor output);
}

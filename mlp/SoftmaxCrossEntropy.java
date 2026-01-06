package mlp;

public class SoftmaxCrossEntropy implements Op {

    public Tensor forward(Tensor logits, int label) {
        int n = logits.size;
        double[] probs = new double[n];

        //  数值稳定：减 max
        double max = logits.data[0];
        for (int i = 1; i < n; i++) {
            if (logits.data[i] > max) max = logits.data[i];
        }

        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            probs[i] = Math.exp(logits.data[i] - max);
            sum += probs[i];
        }

        for (int i = 0; i < n; i++) {
            probs[i] /= sum;
        }

        //  loss
        double lossVal = -Math.log(probs[label]);

        Tensor loss = new Tensor(new double[]{lossVal}, new int[]{1}, true);

        // 构建计算图
        loss.creator = this;
        loss.parents = new Tensor[]{logits};

        // ctx 保存 softmax 概率 + label
        loss.ctx = new Object[]{probs, label};

        return loss;
    }

    @Override
    public void backward(Tensor loss) {
        Tensor logits = loss.parents[0];
        Object[] ctx = (Object[]) loss.ctx;

        double[] probs = (double[]) ctx[0];
        int label = (int) ctx[1];

        // ∂L/∂z = softmax - one_hot
        for (int i = 0; i < logits.size; i++) {
            double grad = probs[i];
            if (i == label) grad -= 1.0;
            logits.grad[i] += grad;
        }
    }
    
    public int predict(Tensor logits) {
        int n = logits.size;
        double maxValue = -1;
        int maxidx = -1;
        double[] probs = new double[n];

        //  数值稳定：减 max
        double max = logits.data[0];
        for (int i = 1; i < n; i++) {
            if (logits.data[i] > max) max = logits.data[i];
        }

        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            probs[i] = Math.exp(logits.data[i] - max);
            if (probs[i] > maxValue) {
            	maxValue = probs[i];
            	maxidx = i;
            }
        }
        return maxidx;
    }

}


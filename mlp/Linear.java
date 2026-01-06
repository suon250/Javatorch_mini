package mlp;

import java.util.Random;

public class Linear implements Op {

    public Tensor W;   // [out, in]
    public Tensor b;   // [out]

    private int inFeatures;
    private int outFeatures;

    public Linear(int inFeatures, int outFeatures) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;

        // Xavier-like 初始化（简单版）
        double scale = Math.sqrt(2.0 / inFeatures);

        double[] wData = new double[outFeatures * inFeatures];
        double[] bData = new double[outFeatures];

        Random rand = new Random();
        for (int i = 0; i < wData.length; i++) {
            wData[i] = rand.nextGaussian() * scale;
        }

        // bias 初始化为 0
        for (int i = 0; i < bData.length; i++) {
            bData[i] = 0.0;
        }

        W = new Tensor(wData, new int[]{outFeatures, inFeatures}, true);
        b = new Tensor(bData, new int[]{outFeatures}, true);
    }
    
    public Tensor forward(Tensor x) {
        // x: [in]
        double[] out = new double[outFeatures];

        for (int i = 0; i < outFeatures; i++) {
            double sum = b.data[i];
            for (int j = 0; j < inFeatures; j++) {
                sum += W.data[i * inFeatures + j] * x.data[j];
            }
            out[i] = sum;
        }

        Tensor y = new Tensor(out, new int[]{outFeatures}, true);

        // 构建计算图
        y.creator = this;
        y.parents = new Tensor[]{x, W, b};

        // cache 前向输入，给 backward 用
        y.ctx = x;

        return y;
    }
    
    @Override
    public void backward(Tensor y) {
        Tensor x = (Tensor) y.ctx;
        double[] gradOut = y.grad; // ∂L/∂y

        // 对 x 的梯度：dx = W^T * gradOut
        if (x.requiresGrad == true) {
        	for (int j = 0; j < inFeatures; j++) {
                double sum = 0.0;
                for (int i = 0; i < outFeatures; i++) {
                    sum += W.data[i * inFeatures + j] * gradOut[i];
                }
                x.grad[j] += sum;
            }
        }

        // 对 W 的梯度：dW = gradOut ⊗ x
        for (int i = 0; i < outFeatures; i++) {
            for (int j = 0; j < inFeatures; j++) {
                W.grad[i * inFeatures + j]
                        += gradOut[i] * x.data[j];
            }
        }

        // 对 b 的梯度
        for (int i = 0; i < outFeatures; i++) {
            b.grad[i] += gradOut[i];
        }
    }
    
    // simple optimizer
    public void step(double lr) {
        for (int i = 0; i < W.data.length; i++) {
            W.data[i] -= lr * W.grad[i];
        }
        for (int i = 0; i < b.data.length; i++) {
            b.data[i] -= lr * b.grad[i];
        }
    }

}



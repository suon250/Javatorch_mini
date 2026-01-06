package mlp;

public class TrainMNIST {

    public static void main(String[] args) throws Exception {

        // 读 MNIST训练集
        double[][][] images = MNISTLoader.loadTrainImages();
        int[] labels = MNISTLoader.loadTrainLabels();
        // 读取MNIST测试集
        double[][][] testImages = MNISTLoader.loadTestImages();
        int[] testLabels = MNISTLoader.loadTestLabels();
        
        //️⃣ 模型
        Linear fc1 = new Linear(784, 128);
        ReLU relu = new ReLU();
        Linear fc2 = new Linear(128, 10);

        SoftmaxCrossEntropy lossFn = new SoftmaxCrossEntropy();

        double lr = 0.01;

        // 训练
        for (int epoch = 0; epoch < 10; epoch++) {
            double totalLoss = 0.0;

            for (int i = 0; i < images.length; i++) {

                // ---- 数据 ----
                double[] xData = MNISTLoader.flatten(images[i]);
                Tensor x = new Tensor(xData, new int[]{784}, false);

                // ---- forward ----
                Tensor h = relu.forward(fc1.forward(x));
                Tensor logits = fc2.forward(h);
                Tensor loss = lossFn.forward(logits, labels[i]);
                // System.out.println(loss.data[0]);

                totalLoss += loss.data[0];

                // ---- backward ----
                fc1.W.zeroGrad();
                fc1.b.zeroGrad();
                fc2.W.zeroGrad();
                fc2.b.zeroGrad();

                loss.backward();

                // ---- SGD ----
                fc1.step(lr);
                fc2.step(lr);
            }

            System.out.println("Epoch " + epoch +
                    " | loss = " + totalLoss / images.length);
        }
        
        // 测试
        int cnt = 0;
        for (int i = 0; i < images.length; i++) {
        	int predict = -1;
            // ---- 数据 ----
            double[] xData = MNISTLoader.flatten(images[i]);
            Tensor x = new Tensor(xData, new int[]{784}, false);

            // ---- forward ----
            Tensor h = relu.forward(fc1.forward(x));
            Tensor logits = fc2.forward(h);
            predict = lossFn.predict(logits);
            
            if (predict == labels[i]) cnt++;
            
        }
        double accuracy = 100.0 * cnt / images.length;
        System.out.println("Model accuracy: " + accuracy + "%");
    }
}


package mlp;

import java.io.IOException;

public class Driver {

	public static void main(String[] args) {
		int[] trainLabels = null;
		double[][][] trainImages = null;
		int id = 0; // choose the first sample
		int sampleLabel;
		double[][] sampleFeature;
		
		// load TrainImages dataset
		try {
			trainImages = MNISTLoader.loadTrainImages();
			trainLabels = MNISTLoader.loadTrainLabels();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// obtain one sample images
		sampleLabel = trainLabels[id];
		sampleFeature = trainImages[id];
		double[] flatSample = MNISTLoader.flatten(sampleFeature);
		System.out.println("the first image after flattening:");
		for (int i = 0; i < 10; i++) {
			System.out.print(sampleFeature[0][i]);
			System.out.print(" ");
		}
		System.out.println();
		System.out.println();
		
		// initalize the image as Tensor
		Tensor sampleTensor = new Tensor(flatSample, new int[]{28, 28}, true);
		double[] sampleData = sampleTensor.data;
		System.out.println("the first image in Tensor:");
		for (int i = 0; i < 10; i++) {
			System.out.print(sampleData[i]);
			System.out.print(" ");
		}
		
		

	}

}

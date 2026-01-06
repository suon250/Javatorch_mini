package mlp;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MNISTLoader {
    // the position of the MNIST dataset relative to the root of the project
    public static final String TRAIN_IMAGES = "data/train-images.idx3-ubyte";
    public static final String TRAIN_LABELS = "data/train-labels.idx1-ubyte";
    public static final String TEST_IMAGES = "data/t10k-images.idx3-ubyte";
    public static final String TEST_LABELS = "data/t10k-labels.idx1-ubyte";
    
    // read the images(as double)
    // IOEcxeption might occur while loading the dataset
    public static double[][][] loadMNISTImages(String file) throws IOException {
        DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(file)));

        int magic = dis.readInt();
        int numImages = dis.readInt();
        int numRows = dis.readInt();
        int numCols = dis.readInt();

        double[][][] images = new double[numImages][numRows][numCols];

        for (int i = 0; i < numImages; i++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    int pixel = dis.readUnsignedByte();
                    images[i][r][c] = pixel / 255.0; // normalization
                }
            }
        }

        dis.close();
        return images;
    }
    
    // load the labels
    public static int[] loadMNISTLabels(String file) throws IOException {
        DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(file)));
        
        int magic = dis.readInt();
        int numLabels = dis.readInt();
        
        int[] labels = new int[numLabels];
        
        for (int i = 0; i < numLabels; i++) {
            labels[i] = dis.readUnsignedByte();
        }
        
        dis.close();
        return labels;
    }
    
    // add convenient method
    public static double[][][] loadTrainImages() throws IOException {
        return loadMNISTImages(TRAIN_IMAGES);
    }
    
    public static double[][][] loadTestImages() throws IOException {
        return loadMNISTImages(TEST_IMAGES);
    }
    
    public static int[] loadTrainLabels() throws IOException {
        return loadMNISTLabels(TRAIN_LABELS);
    }
    
    public static int[] loadTestLabels() throws IOException {
        return loadMNISTLabels(TEST_LABELS);
    }
    
    // flatten the image
    public static double[] flatten(double[][] img) {
        double[] flat = new double[28 * 28];
        int idx = 0;
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                flat[idx++] = img[i][j];
            }
        }
        return flat;
    }
    
    // visualize the image
    public static void showImage(double[][] img) {
    	for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                char ch = img[r][c] > 0.5 ? '#' : '.';
                System.out.print(ch);
            }
            System.out.println();
        }
    }
    
    // test method
    public static void main(String[] args) {
        try {
            System.out.println("Testing MNISTLoader...");
            
            // load a sample image and labels for testing
            System.out.println("Loading MNIST data...");
            
            double[][][] trainImages = loadTrainImages();
            int[] trainLabels = loadTrainLabels();
            
            System.out.println("Successfully loaded " + trainImages.length + " training images");
            System.out.println("Successfully loaded " + trainLabels.length + " training labels");
            
            // show the basic information fo the first label
            System.out.println("\nFirst image:");
            System.out.println("Label: " + trainLabels[0]);
            System.out.println("Image shape: " + trainImages[0].length + "x" + trainImages[0][0].length);
            
            // show the first image 28 * 28, with number 5 (in pixels)
            System.out.println("\nASCII preview:");
            showImage(trainImages[0]);
            
        } catch (IOException e) {
            System.err.println("Error loading MNIST data!");
            e.printStackTrace();
            System.err.println("\nTroubleshooting tips:");
            System.err.println("1. Make sure you have the MNIST files in the 'data/' folder");
            System.err.println("2. Check the file names: train-images.idx3-ubyte, train-labels.idx1-ubyte, etc.");
            System.err.println("3. Refresh Eclipse project (F5) after adding files");
            System.err.println("4. Files should be in: " + System.getProperty("user.dir") + "/data/");
        }
    }
}
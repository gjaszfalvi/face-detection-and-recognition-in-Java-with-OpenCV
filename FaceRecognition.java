
import java.io.File;
import java.io.FilenameFilter;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.util.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_objdetect;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_face.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

/**
 * FaceRecognition.java
 * The app recognizes an image of a face by comparing it to a database of faces
 * At this stage the face image should be 100x100, and the testimage should be the same size
 * The database image filename starts with the label, then the name of the person, 
 * then number of the image separated by hyphen, i.e.:
 * 1-max-1.jpg
 * 
 * Based on:http://pcbje.com/2012/12/doing-face-recognition-with-javacv/
 * By Petter Christian Bjelland
 * 
 * @author Gabor Jaszfalvi
 * 2016
 */

public class FaceRecognition {
    public static void main(String[] args) {
        // loading openCV
        Loader.load(opencv_objdetect.class);
        // set image database
        String trainingDir = ("/Users/shop/NetBeansProjects/OpenCV/src/100x100");
        // set image for testing face recognition
        Mat testImage = imread(FaceRecognition.class.getResource("006.jpg").getPath(), CV_LOAD_IMAGE_GRAYSCALE);
        
        // declare variable to filter imagefiles in database folder (useful if contains other files)
        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };
        // create array of database imagefiles for comparison
        File[] imageFiles = new File(trainingDir).listFiles(imgFilter);
        
        // declare vectors that holds the database images and their labels
        MatVector images = new MatVector(imageFiles.length);
        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        // create a buffer to store labels
        IntBuffer labelsBuf = labels.createBuffer();
        
        // declare set to store the person's name from database image files )
        // ********* this requires that each person's name must be different in database ************
        Set theSet = new LinkedHashSet();
        // declare a simple counter for for loop
        int counter = 0;
        
        for (File image : imageFiles) {
            // try to read the images from data image folder and put them into the images matrix
            try {
            Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            images.put(counter, img);
            } catch (Exception ex) {
            ex.printStackTrace();}
            
            // get the labels and names of data image files
            int label = Integer.parseInt(image.getName().split("\\-")[0]);
            String name = (image.getName().split("\\-")[1]);
            // add the names to the set
            theSet.add(name);
            // put the labels in buffer
            labelsBuf.put(counter, label);
            counter++;
        }
        
        // declare the facerecognizer (type of Fisher, Eigen or Local Binary Pattern Histogram)
        //FaceRecognizer faceRecognizer = createFisherFaceRecognizer();
        //FaceRecognizer faceRecognizer = createEigenFaceRecognizer();
        FaceRecognizer faceRecognizer = createLBPHFaceRecognizer();
        
        // create an array of names from the set to identify the person by label and filename
        int sizeofset = theSet.size();
        String[] theList = new String[sizeofset];
        Iterator it = theSet.iterator();
        for (int i =0; i<sizeofset;i++) {
            theList[i] = (String)it.next();
        }
        // train the recognizer (start comparison)
        faceRecognizer.train(images, labels);
        // declare variable to present the result of recognition
        int predictedLabel = faceRecognizer.predict(testImage);
        
        System.out.println("Person identified: " + theList[predictedLabel-1].toUpperCase());
        
    }
}
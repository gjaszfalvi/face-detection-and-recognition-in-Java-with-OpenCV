
import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Set;
import org.bytedeco.javacpp.Loader;
import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.CvScalar;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.CvSize;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_core.cvClearMemStorage;
import static org.bytedeco.javacpp.opencv_core.cvCopy;
import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvFlip;
import static org.bytedeco.javacpp.opencv_core.cvGetSeqElem;
import static org.bytedeco.javacpp.opencv_core.cvGetSize;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.cvPoint;
import static org.bytedeco.javacpp.opencv_core.cvReleaseImage;
import static org.bytedeco.javacpp.opencv_core.cvSetImageROI;
import static org.bytedeco.javacpp.opencv_core.cvSize;
import org.bytedeco.javacpp.opencv_face;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.CV_AA;
import static org.bytedeco.javacpp.opencv_imgproc.CV_FONT_HERSHEY_SIMPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RGB2GRAY;
import org.bytedeco.javacpp.opencv_imgproc.CvFont;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.cvInitFont;
import static org.bytedeco.javacpp.opencv_imgproc.cvRectangle;
import static org.bytedeco.javacpp.opencv_imgproc.cvResize;
import org.bytedeco.javacpp.opencv_objdetect;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_SCALE_IMAGE;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import static org.bytedeco.javacpp.opencv_objdetect.cvReleaseHaarClassifierCascade;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_face.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

public class CapImage {
    public static final String XML_FILE = "haarcascade_frontalface_alt.xml";

    public static void main(String[] args) throws InterruptedException {
        
        // loading openCV
        Loader.load(opencv_objdetect.class);
        // set image database
        String trainingDir = ("/Users/shop/NetBeansProjects/OpenCV/src/100x100");
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
        opencv_core.MatVector images = new opencv_core.MatVector(imageFiles.length);
        opencv_core.Mat labels = new opencv_core.Mat(imageFiles.length, 1, CV_32SC1);
        // create a buffer to store labels
        IntBuffer labelsBuf = labels.createBuffer();
        
        // declare set to store the person's name from database image files )
        // ********* this requires that each person's name must be different in database ************
        Set theSet = new LinkedHashSet();
        // declare a simple counter for for loop
        int counter = 0;
        
        
        for (File image : imageFiles) {
            // try to read the images from data image folder and put them into the images matrix
            opencv_core.Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            images.put(counter, img);
            
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
        opencv_face.FaceRecognizer faceRecognizer = createLBPHFaceRecognizer();
        
        // create an array of names from the set to identify the person by label and filename
        int sizeofset = theSet.size();
        String[] theList = new String[sizeofset];
        Iterator it = theSet.iterator();
        for (int i =0; i<sizeofset;i++) {
            theList[i] = (String)it.next();
        }
        // train the recognizer (start comparison)
        faceRecognizer.train(images, labels);
        
        
        // Declare img as IplImage
        IplImage img;
        //CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(XML_FILE));
        CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(Demo.class.getResource("haarcascade_frontalface_alt.xml").getPath()));

        CvMemStorage storage = CvMemStorage.create();
        // Create canvas frame for displaying webcam.
        CanvasFrame canvas = new CanvasFrame("Webcam");

        // Set Canvas frame to close on exit
        canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);

        // Declare FrameGrabber to import output from webcam
        FrameGrabber grabber = new OpenCVFrameGrabber("");
        OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();

        try {

            // Start grabber to capture video
            
            while (true) {

                // inser grabed video fram to IplImage img
            grabber.start();
            img = converter.convert(grabber.grab());
            // Set canvas size as per dimentions of video frame.
            canvas.setCanvasSize(640,400);
                if (img != null) {
                    // Flip image horizontally
                    //cvFlip(img, img, 1);
                    // Detect Face
                    CvSeq faces = cvHaarDetectObjects(img, cascade, storage, 1.2, 3, CV_HAAR_SCALE_IMAGE);

                    cvClearMemStorage(storage);
                    int total_Faces = faces.total();

                    /*
                    *   1. Draw a rectangle around detected face
                    *   2. Crop the face
                    *   3. Resize the cropped face to 100x100 for recognition comparison
                    */
                    for (int i = 0; i < total_Faces; i++) {
                            CvRect r = new CvRect(cvGetSeqElem(faces, i));
                            cvRectangle(img, cvPoint(r.x(), r.y()),cvPoint(r.width() + r.x(), r.height() + r.y()),CvScalar.GREEN, 2, CV_AA, 0);
                            cvSetImageROI(img, r);
                            IplImage cropped = cvCreateImage(cvGetSize(img), img.depth(), img.nChannels());
                            // Copy original image (only ROI) to the cropped image
                            cvCopy(img, cropped);
                            // resize cropped image
                            IplImage resizedImage = cvCreateImage(cvSize(100,100), cropped.depth(), cropped.nChannels());
                            cvResize(cropped, resizedImage);
                            //cvSaveImage("cropped.png",resizedImage);
                            
                            IplImage grey = cvCreateImage(cvGetSize(resizedImage), 8, 1);
                            cvCvtColor(resizedImage, grey, CV_RGB2GRAY);
                            //cvSaveImage("croppeded.png",grey);

                            Mat testImage = new Mat(grey);
                            // declare variable to present the result of recognition
                            int predictedLabel = faceRecognizer.predict(testImage);
                            String result = theList[predictedLabel-1].toUpperCase();
                            int pos_x = Math.max(r.x() - 10, 0);
                            int pos_y = Math.max(r.y() - 10, 0);
                            //System.out.println("Person identified: " + result);
                            CvFont myfont = new CvFont();
                            cvInitFont(myfont, CV_FONT_HERSHEY_SIMPLEX, 2.5, 2.5,0.0,3,8); //rate of 
                            cvPutText(img,result,cvPoint(10,100),myfont,CvScalar.GREEN);
    
                            cvReleaseImage(resizedImage);

                    }
                    // Show video frame in canvas
                    Frame frame = converter.convert(img);
                    canvas.showImage(frame);
                    //canvas.repaint();
                    Thread.sleep(50);
                }   
            }
        } catch (Exception e) {
            e.printStackTrace();
        }           
    }
}
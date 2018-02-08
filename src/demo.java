import org.opencv.core.*;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import static org.opencv.core.Core.getTickFrequency;

class Runner {
    public void run() {
        int inWidth = 300;
        int inHeight = 300;
        Size inSize = new Size(inWidth, inHeight);
        Scalar inMean = new Scalar(104.0, 177.0, 123.0);
        float confThreshold = 0.5f;

        String prototxt = "face_detector/deploy.prototxt";
        String caffemodel = "face_detector/res10_300x300_ssd_iter_140000.caffemodel";

        Mat image = Imgcodecs.imread("data/test.jpg");
        VideoCapture cap = new VideoCapture(0);
        Mat frame =  new Mat();
        Imshow ims = new Imshow("demo");
        Net net = Dnn.readNetFromCaffe(prototxt, caffemodel);
        net.setPreferableTarget(Dnn.DNN_TARGET_CPU);
        MatOfDouble timings = new MatOfDouble();

        for(;;){
            cap.read(frame);

            int cols = frame.cols();
            int rows = frame.rows();

            Mat blob = Dnn.blobFromImage(frame, 1.0, inSize, inMean, false, false);
            net.setInput(blob);
            Mat detections = net.forward();

            long perf_stats = net.getPerfProfile(timings);
            System.out.printf("Inference time, ms: %.2f\n", (perf_stats / getTickFrequency() * 1000));

            detections = detections.reshape(1, (int)detections.total() / 7);

            System.out.println(detections.size());
            System.out.println(detections.get(0, 0));

            for (int i = 0; i < detections.rows(); ++i) {
                double confidence = detections.get(i, 2)[0];
                if (confidence > confThreshold) {
//                    int classId = (int)detections.get(i, 1)[0];
                    int xLeftBottom = (int)(detections.get(i, 3)[0] * cols);
                    int yLeftBottom = (int)(detections.get(i, 4)[0] * rows);
                    int xRightTop   = (int)(detections.get(i, 5)[0] * cols);
                    int yRightTop   = (int)(detections.get(i, 6)[0] * rows);
                    // Draw rectangle around detected object.
                    Imgproc.rectangle(frame, new Point(xLeftBottom, yLeftBottom),
                            new Point(xRightTop, yRightTop),
                            new Scalar(0, 255, 0));
                    String label = String.format("face: %.4f", confidence);
                    int[] baseLine = new int[1];
                    Size labelSize = Imgproc.getTextSize(label, Core.FONT_HERSHEY_SIMPLEX, 0.5,
                            1, baseLine);
                    // Draw background for label.
                    Imgproc.rectangle(frame, new Point(xLeftBottom, yLeftBottom - labelSize.height),
                            new Point(xLeftBottom + labelSize.width, yLeftBottom + baseLine[0]),
                            new Scalar(255, 255, 255), Core.FILLED);
                    // Write class name and confidence.
                    Imgproc.putText(frame, label, new Point(xLeftBottom, yLeftBottom),
                            Core.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 0));
                }
            }

            if (ims.CloseTag == true)
                break;
            else
                ims.showImage(frame);
        }
        ims.release();
        cap.release();
        System.out.println("done!");
    }
}

public class demo {

    public static void main(String[] args){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new Runner().run();
    }
}
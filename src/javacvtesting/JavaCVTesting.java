/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package javacvtesting;

import java.awt.FlowLayout;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;

/**
 *
 * @author sonng
 */
public class JavaCVTesting {

    /**
     * @param args the command line arguments
     *//*
  */



    public static BufferedImage convert(Mat m){
        Mat image_tmp = m;

        MatOfByte matOfByte = new MatOfByte();

        Imgcodecs.imencode(".jpg", image_tmp, matOfByte); 

        byte[] byteArray = matOfByte.toArray();
        BufferedImage bufImage = null;

        try {

            InputStream in = new ByteArrayInputStream(byteArray);
            bufImage = ImageIO.read(in);

        } catch (Exception e) {
            e.printStackTrace();
        }finally{
            return bufImage;
        }
    }
    public static Mat convert(BufferedImage i){
        BufferedImage image = i;
        byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(image.getHeight(),image.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, data);
        return mat;
    }
    public static void show(BufferedImage i){
        JFrame frame = new JFrame();
        frame.getContentPane().setLayout(new FlowLayout());
        frame.getContentPane().add(new JLabel(new ImageIcon(i)));
        frame.pack();
        frame.setVisible(true);
    }
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat src = Imgcodecs.imread("duc.jpg");
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2HSV);
        Mat dest = new Mat();
      // Mat dest = new Mat(src.width(), src.height(), src.type());
        Core.inRange(src, new Scalar(58,125,0), new Scalar(256,256,256), dest);

        Mat erode = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
        Mat dilate = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5));
        Imgproc.erode(dest, dest, erode);
        Imgproc.erode(dest, dest, erode);

        Imgproc.dilate(dest, dest, dilate);
        Imgproc.dilate(dest, dest, dilate);

        List<MatOfPoint> contours = new ArrayList<>();

        Imgproc.findContours(dest, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(dest, contours, -1, new Scalar(255,255,0));

        Panel p = new Panel();
//        p.setImage(convert(dest));
        p.show();
    }
}

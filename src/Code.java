
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author sonng
 */
public class Code {

    public void Histogram() {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            Mat source = Imgcodecs.imread("duc.jpg",
                    Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Mat destination = new Mat(source.rows(), source.cols(), source.type());

            Imgproc.equalizeHist(source, destination);
            Imgcodecs.imwrite("ducHist.jpg", destination);
            LoadImg("ducHist.jpg");

        } catch (Exception e) {
            System.out.println("error: " + e.getMessage());
        }
    }

    public void Gamma(double g) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat im = Imgcodecs.imread("duc.jpg");

        double gamma = g;
        Mat lut = new Mat(1, 256, CvType.CV_8UC1);
        lut.setTo(new Scalar(0));

        for (int i = 0; i < 256; i++) {
            lut.put(0, i, Math.pow((double) (1.0 * i / 255), gamma) * 255);
        }
        Core.LUT(im, lut, im);
        Imgcodecs.imwrite("ducGamma" + g + ".jpg", im);
        LoadImg("ducGamma" + g + ".jpg");

    }

    public void Gauss(int size) {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            Mat source = Imgcodecs.imread("ducOstu.jpg",
                    Imgcodecs.CV_LOAD_IMAGE_COLOR);

            Mat destination = new Mat(source.rows(), source.cols(), source.type());
            Imgproc.GaussianBlur(source, destination, new Size(size, size), 0);
            Imgcodecs.imwrite("Gauss" + size + ".jpg", destination);
            LoadImg("Gauss" + size + ".jpg");
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    public void Moyen(int n) {
        try {
            // Initialiser la taille du filtre et télécharger la bibliothèque
            int kernelSize = n;
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            // lire les données d'image
            Mat source = Imgcodecs.imread("duc.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Mat destination = new Mat(source.rows(), source.cols(), source.type());
            Mat kernel = Mat.ones(kernelSize, kernelSize, CvType.CV_32F);
            // Parcourir chaque élément de la matrice en lignes, colonnes
            for (int i = 0; i < kernel.rows(); i++) {
                for (int j = 0; j < kernel.cols(); j++) {
                    double[] m = kernel.get(i, j); // nombre de composant matriciels
                    for (int k = 0; k < m.length; k++) {
                        // effectuer scission la moyenne pour chaque élément matrice
                        m[k] = m[k] / (kernelSize * kernelSize);
                    }
                    kernel.put(i, j, m);
                }
            }
            // Effectuez le filtre coulissant. Avec une valeur de -1, 
            //la taille de l'image est identique à celle de l'image originale
            Imgproc.filter2D(source, destination, -1, kernel);
            Imgcodecs.imwrite("Moyen" + n + ".jpg", destination);
            LoadImg("Moyen" + n + ".jpg");
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    public void Median(int n) {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            Mat source = Imgcodecs.imread("Adaptivemean_thresh_binary.jpg",
                    Imgcodecs.CV_LOAD_IMAGE_COLOR);
            Mat destination = new Mat(source.rows(), source.cols(), source.type());
            Imgproc.medianBlur(source, destination, n);
            Imgcodecs.imwrite("ducMedian" + n + ".jpg", destination);
            LoadImg("ducMedian" + n + ".jpg");
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    public void Prewitt(int n) {
        try {
            int kernelSize = 9;
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            Mat source = Imgcodecs.imread("duc.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Mat destination = new Mat(source.rows(), source.cols(), source.type());
            Mat kernel = null;
            // dọc
            if (n == 0) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, -1);
                        put(0, 1, -1);
                        put(0, 2, -1);

                        put(1, 0, 0);
                        put(1, 1, 0);
                        put(1, 2, 0);

                        put(2, 0, 1);
                        put(2, 1, 1);
                        put(2, 2, 1);
                    }
                };
                //ngang
            } else if (n == 1) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, -1);
                        put(0, 1, 0);
                        put(0, 2, 1);

                        put(1, 0, -1);
                        put(1, 1, 0);
                        put(1, 2, 1);

                        put(2, 0, -1);
                        put(2, 1, 0);
                        put(2, 2, 1);
                        //x+y 
                    }
                };
            } else if (n == 2) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, -2);
                        put(0, 1, -1);
                        put(0, 2, 0);

                        put(1, 0, -1);
                        put(1, 1, 0);
                        put(1, 2, 2);

                        put(2, 0, 0);
                        put(2, 1, 1);
                        put(2, 2, 2);
                    }
                };
            }

            Imgproc.filter2D(source, destination, -1, kernel);
            double x = Math.random();

            Imgcodecs.imwrite("Prewitt" + x + ".jpg", destination);
            LoadImg("Prewitt" + x + ".jpg");
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }

    }

    public void Sobel(int n) {
        try {
            int kernelSize = 9;
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            Mat source = Imgcodecs.imread("duc.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Mat destination = new Mat(source.rows(), source.cols(), source.type());
            Mat kernel = null;
            // dọc
            if (n == 0) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, -1);
                        put(0, 1, -2);
                        put(0, 2, -1);

                        put(1, 0, 0);
                        put(1, 1, 0);
                        put(1, 2, 0);

                        put(2, 0, 1);
                        put(2, 1, 2);
                        put(2, 2, 1);
                    }
                };
                //ngang
            } else if (n == 1) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, -1);
                        put(0, 1, 0);
                        put(0, 2, 1);

                        put(1, 0, -2);
                        put(1, 1, 0);
                        put(1, 2, 2);

                        put(2, 0, -1);
                        put(2, 1, 0);
                        put(2, 2, 1);
                    }
                };
                //x+y 
            } else if (n == 2) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, -2);
                        put(0, 1, -2);
                        put(0, 2, 0);

                        put(1, 0, -2);
                        put(1, 1, 0);
                        put(1, 2, 2);

                        put(2, 0, 0);
                        put(2, 1, 2);
                        put(2, 2, 2);
                    }
                };
            }

            Imgproc.filter2D(source, destination, -1, kernel);
            double x = Math.random();
            Imgcodecs.imwrite("Sobel" + x + ".jpg", destination);
            LoadImg("Sobel" + x + ".jpg");
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    public void Robinson(int n) {
        try {
            int kernelSize = 9;
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            Mat source = Imgcodecs.imread("duc.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Mat destination = new Mat(source.rows(), source.cols(), source.type());
            Mat kernel = null;
            // dọc
            if (n == 0) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, -1);
                        put(0, 1, -2);
                        put(0, 2, -1);

                        put(1, 0, 0);
                        put(1, 1, 0);
                        put(1, 2, 0);

                        put(2, 0, 1);
                        put(2, 1, 2);
                        put(2, 2, 1);
                    }
                };
                //ngang
            } else if (n == 1) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, -1);
                        put(0, 1, 0);
                        put(0, 2, 1);

                        put(1, 0, -2);
                        put(1, 1, 0);
                        put(1, 2, 2);

                        put(2, 0, -1);
                        put(2, 1, 0);
                        put(2, 2, 1);
                    }
                };
                //x+y 
            } else if (n == 2) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, -2);
                        put(0, 1, -2);
                        put(0, 2, 0);

                        put(1, 0, -2);
                        put(1, 1, 0);
                        put(1, 2, 2);

                        put(2, 0, 0);
                        put(2, 1, 2);
                        put(2, 2, 2);
                    }
                };
            }

            Imgproc.filter2D(source, destination, -1, kernel);
            double x = Math.random();
            Imgcodecs.imwrite("Roberst" + x + ".jpg", destination);
            LoadImg("Roberst" + x + ".jpg");
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    public void Laplace(int n) {
        try {
            int kernelSize = 9;
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            Mat source = Imgcodecs.imread("duc.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Mat destination = new Mat(source.rows(), source.cols(), source.type());
            Mat kernel = null;
            // dọc
            if (n == 0) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, 0);
                        put(0, 1, -1);
                        put(0, 2, 0);

                        put(1, 0, -1);
                        put(1, 1, 4);
                        put(1, 2, -1);

                        put(2, 0, 0);
                        put(2, 1, -1);
                        put(2, 2, 0);
                    }
                };
                //ngang
            } else if (n == 1) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, 0);
                        put(0, 1, 1);
                        put(0, 2, 0);

                        put(1, 0, 1);
                        put(1, 1, -4);
                        put(1, 2, 1);

                        put(2, 0, 0);
                        put(2, 1, 1);
                        put(2, 2, 0);
                    }
                };
                //x+y 
            } else if (n == 2) {
                kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F) {
                    {
                        put(0, 0, 0);
                        put(0, 1, 0);
                        put(0, 2, 0);

                        put(1, 0, 0);
                        put(1, 1, 0);
                        put(1, 2, 0);

                        put(2, 0, 0);
                        put(2, 1, 0);
                        put(2, 2, 0);
                    }
                };
            }

            Imgproc.filter2D(source, destination, -1, kernel);
            double x = Math.random();
            Imgcodecs.imwrite("Laplace" + x + ".jpg", destination);
            LoadImg("Laplace" + x + ".jpg");
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    public void Ostu() {
        try {
//            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//            Mat source = Imgcodecs.imread("watercoins.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
//            Mat destination = new Mat(source.rows(), source.cols(), source.type());
//            destination = source;
//            Imgproc.threshold(source, destination, 125, 255, Imgproc.THRESH_OTSU);
//            Imgcodecs.imwrite("ducOstu.jpg", destination);
            // Loading the OpenCV core library
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            String file = "distnceTransform.jpg";
            Mat src = Imgcodecs.imread(file);
            Mat dst = new Mat();
            Imgproc.threshold(src, dst, 200, 255, Imgproc.THRESH_BINARY_INV);
            Imgcodecs.imwrite("ducOstu.jpg", dst);
            System.out.println("Image Processed");
            LoadImg("ducOstu.jpg");
        } catch (Exception e) {
            System.out.println("error: " + e.getMessage());
        }

    }

    public void FourierTranform() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        String filename = "duc.jpg";
        Mat I = Imgcodecs.imread(filename, Imgcodecs.IMREAD_GRAYSCALE);
        if (I.empty()) {
            System.out.println("Error opening image");
            System.exit(-1);
        }
        Mat padded = new Mat();                     //expand input image to optimal size
        int m = Core.getOptimalDFTSize(I.rows());
        int n = Core.getOptimalDFTSize(I.cols()); // on the border add zero values
        Core.copyMakeBorder(I, padded, 0, m - I.rows(), 0, n - I.cols(), Core.BORDER_CONSTANT, Scalar.all(0));
        List<Mat> planes = new ArrayList<Mat>();
        padded.convertTo(padded, CvType.CV_32F);
        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
        Mat complexI = new Mat();
        Core.merge(planes, complexI);         // Add to the expanded another plane with zeros
        Core.dft(complexI, complexI);         // this way the result may fit in the source matrix
        // compute the magnitude and switch to logarithmic scale
        // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
        Core.split(complexI, planes);                               // planes.get(0) = Re(DFT(I)
        // planes.get(1) = Im(DFT(I))
        Core.magnitude(planes.get(0), planes.get(1), planes.get(0));// planes.get(0) = magnitude
        Mat magI = planes.get(0);
        Mat matOfOnes = Mat.ones(magI.size(), magI.type());
        Core.add(matOfOnes, magI, magI);         // switch to logarithmic scale
        Core.log(magI, magI);
        // crop the spectrum, if it has an odd number of rows or columns
        magI = magI.submat(new Rect(0, 0, magI.cols() & -2, magI.rows() & -2));
        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        int cx = magI.cols() / 2;
        int cy = magI.rows() / 2;
        Mat q0 = new Mat(magI, new Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        Mat q1 = new Mat(magI, new Rect(cx, 0, cx, cy));  // Top-Right
        Mat q2 = new Mat(magI, new Rect(0, cy, cx, cy));  // Bottom-Left
        Mat q3 = new Mat(magI, new Rect(cx, cy, cx, cy)); // Bottom-Right
        Mat tmp = new Mat();               // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
        magI.convertTo(magI, CvType.CV_8UC1);
        Core.normalize(magI, magI, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1); // Transform the matrix with float values
        // into a viewable image form (float between
        // values 0 and 255).
        Imgcodecs.imwrite("hinhanh1.jpg", I);
        Imgcodecs.imwrite("hinhanh2.jpg", magI);
        LoadImg("hinhanh2.jpg");
    }

    public void Watershed() {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            Mat image = new Mat();
            image = Imgcodecs.imread("watercoins.jpg");

            Mat binaryImage = new Mat();
            Imgproc.cvtColor(image, binaryImage, Imgproc.COLOR_BGR2GRAY);
            Imgproc.threshold(binaryImage, binaryImage, 0, 255, Imgproc.THRESH_OTSU);

            // vẽ nền
            Mat fg = new Mat(image.size(), CvType.CV_8U);
            Imgproc.erode(binaryImage, fg, new Mat(), new Point(-1, -1), 1);

            // vẽ biên
            Mat bg = new Mat(image.size(), CvType.CV_8U);
            Imgproc.dilate(binaryImage, bg, new Mat(), new Point(-1, -1), 0);
            Imgproc.threshold(bg, bg, 1, 128, Imgproc.THRESH_BINARY_INV);

            Mat markers = new Mat(image.size(), CvType.CV_8U, new Scalar(0));
            Core.add(fg, bg, markers);

            setMarkers(markers);
            Mat result = process(image);

            double x = Math.random();
            Imgcodecs.imwrite("Watershed.jpg", result);
            LoadImg("Watershed.jpg");
        } catch (Exception e) {
        }
    }
    public Mat markers;

    public void setMarkers(Mat markerImage) {
        markers = new Mat();
        markerImage.convertTo(markers, CvType.CV_32S);
    }

    public Mat process(Mat image) {
        Imgproc.watershed(image, markers);
        markers.convertTo(markers, CvType.CV_8U);
        return markers;
    }

    public void BinaryImage() {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            Mat source = Imgcodecs.imread("test.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Mat destination = new Mat(source.rows(), source.cols(), source.type());

            destination = source;

            int erosion_size = 15;
            int dilation_size = 10;

            Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * erosion_size + 1, 2 * erosion_size + 1));
            Imgproc.erode(source, destination, element);
            Imgcodecs.imwrite("BinaryImageErosion.jpg", destination);
            
            Mat source2 = Imgcodecs.imread("test.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Mat destination2 = new Mat(source2.rows(), source2.cols(), source2.type());
            destination2 = source2;
            Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1+ dilation_size , 1+ dilation_size ));
            Imgproc.dilate(source2, destination2, element1);
            Imgcodecs.imwrite("BinaryImageDilation.jpg", destination2);
            LoadImg("BinaryImageErosion.jpg");
            LoadImg("BinaryImageDilation.jpg");
        } catch (Exception e) {
            System.out.println("error: " + e.getMessage());
        }
    }

    public void FindObjects() {
        // Load the library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // Consider the image for processing
        Mat image = Imgcodecs.imread("test.jpg");
        Mat image2 = Imgcodecs.imread("watercoins.jpg");
        Mat imageHSV = new Mat(image.size(), CvType.CV_8UC4);
        Mat imageBlurr = new Mat(image.size(), CvType.CV_8UC4);
        Mat imageA = new Mat(image.size(), CvType.CV_32F);
        Imgproc.cvtColor(image, imageHSV, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(imageHSV, imageBlurr, new Size(5, 5), 0);
        Imgproc.adaptiveThreshold(imageBlurr, imageA, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 7, 5);
        Imgcodecs.imwrite("duc1.jpg", imageBlurr);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(imageA, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.drawContours(imageA, contours, 1, new Scalar(0, 0, 255));
        Imgcodecs.imwrite("draw.jpg", imageA);

        int x = 0;
        for (int i = 0; i < contours.size(); i++) {
            System.out.println(Imgproc.contourArea(contours.get(i)));
            if (Imgproc.contourArea(contours.get(i)) > 0) {
                Rect rect = Imgproc.boundingRect(contours.get(i));
                System.out.println(rect.height);
                if (rect.height > 0 && i%2==0) {
                    x++;
                    System.out.println("Đây là vật số:" + x / 2);
                    //System.out.println(rect.x +","+rect.y+","+rect.height+","+rect.width);
                    Imgproc.rectangle(image2, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255));

                    Imgproc.putText(
                            image2, // Matrix obj of the image
                            "" + i + "", // Text to be added
                            new Point((rect.x +(rect.x + rect.width))/2, (rect.y + (rect.y + rect.height))/2), // point
                            Core.FONT_HERSHEY_SIMPLEX, // front face
                            0.5, // front scale
                            new Scalar(0, 0, 0), // Scalar object for color
                            2 // Thickness
                    );
                }
            }
        }
         Imgproc.putText(
                            image2, // Matrix obj of the image
                            "Cong Son NGUYEN", // Text to be added
                            new Point(15, 15), // point
                            Core.FONT_HERSHEY_SIMPLEX, // front face
                            0.3, // front scale
                            new Scalar(0, 0, 0), // Scalar object for color
                            2 // Thickness
                    );
        Imgcodecs.imwrite("duc2.jpg", image2);
    }

    public void findContour() {
        // Load the library of openCv
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // Consider the image for processing
        Mat image = Imgcodecs.imread("BinaryImageErosion.jpg", Imgproc.COLOR_BGR2GRAY);
        Mat image2 = Imgcodecs.imread("watercoins.jpg");
        Mat imageHSV = new Mat(image.size(), CvType.CV_8UC4);
        Mat imageBlurr = new Mat(image.size(), CvType.CV_8UC4);
        Mat imageA = new Mat(image.size(), CvType.CV_32F);
        Imgproc.cvtColor(image, imageHSV, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(imageHSV, imageBlurr, new Size(5, 5), 0);
        Imgproc.adaptiveThreshold(imageBlurr, imageA, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 7, 5);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(imageA, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); i++) {
            if (Imgproc.contourArea(contours.get(i)) > 0) {
                Rect rect = Imgproc.boundingRect(contours.get(i));
                if ((rect.height > 0 && rect.height < 500) && (rect.width > 0 && rect.width < 500)) {
                    Imgproc.rectangle(image2, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255));
                    Imgproc.putText(
                            image2, // Matrix obj of the image
                            "" + i + "", // Text to be added
                            new Point(rect.x, rect.y), // point
                            Core.FONT_HERSHEY_SIMPLEX, // front face
                            0.5, // front scale
                            new Scalar(0, 0, 0), // Scalar object for color
                            2 // Thickness
                    );
                }
            }
        }
        Imgcodecs.imwrite("output.png", image2);
    }

    public void test() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat m = Imgcodecs.imread("IMG_0089.jpg", Imgcodecs.CV_LOAD_IMAGE_COLOR);
        Mat hsv = new Mat();
        Mat mask = new Mat();
        Mat dilmask = new Mat();
        Mat fin = new Mat();
        Scalar color = new Scalar(239, 117, 94);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.cvtColor(m, hsv, Imgproc.COLOR_RGB2HSV);
        new LoadImage("IMG_0089.jpg", m);
        Scalar lowerThreshold = new Scalar(120, 100, 100);
        Scalar upperThreshold = new Scalar(179, 255, 255);
        Core.inRange(hsv, lowerThreshold, upperThreshold, mask);
        Imgproc.dilate(mask, dilmask, new Mat());
        Imgproc.findContours(dilmask, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        //Imgproc.drawContours(fin, contours, -1, color, 0);
        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
            if (contours.size() > 5) // Minimum size allowed for consideration
            {
                Imgproc.drawContours(fin, contours, contourIdx, color, 3);
            }
        }
        Imgcodecs.imwrite("s.jpg", fin);
    }

    public void BaiKiemTra() {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // Consider the image for processing
        Mat src = Imgcodecs.imread("watercoins.jpg", Imgcodecs.CV_LOAD_IMAGE_COLOR);

        // Creating an empty matrix to store the result
        Mat dst = new Mat();

        Imgproc.adaptiveThreshold(src, dst, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY, 11, 12);

        // Writing the image
        Imgcodecs.imwrite("Adaptivemean_thresh_binary.jpg", dst);

        System.out.println("Image Processed");

    }

    public void DistanceTransform() {
        // Loading the OpenCV core library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat src = Imgcodecs.imread("ducOstu.jpg", 0);

        // Creating an empty matrix to store the results
        Mat dst = new Mat();
        Mat binary = new Mat();

        // Converting the grayscale image to binary image
        Imgproc.threshold(src, binary, 10, 255, Imgproc.THRESH_BINARY);

        // Applying distance transform
        Imgproc.distanceTransform(src, dst, Imgproc.DIST_MASK_3, 3);

        // Writing the image
        Imgcodecs.imwrite("distnceTransform.jpg", dst);

        System.out.println("Image Processed");
    }

    public void LoadImg(String src) {
        Mat m = Imgcodecs.imread(src, Imgcodecs.CV_LOAD_IMAGE_COLOR);
        LoadImage loadImage = new LoadImage(src, m);
    }

}

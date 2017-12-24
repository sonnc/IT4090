
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
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

            Mat source = Imgcodecs.imread("duc.jpg",
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

            Mat source = Imgcodecs.imread("ducMedian.jpg",
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
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            Mat source = Imgcodecs.imread("duc.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Mat destination = new Mat(source.rows(), source.cols(), source.type());
            destination = source;
            Imgproc.threshold(source, destination, 125, 255, Imgproc.THRESH_OTSU);
            Imgcodecs.imwrite("ducOstu.jpg", destination);
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
            image = Imgcodecs.imread("duc.jpg");

            Mat binaryImage = new Mat();
            Imgproc.cvtColor(image, binaryImage, Imgproc.COLOR_BGR2GRAY);
            Imgproc.threshold(binaryImage, binaryImage, 0, 255, Imgproc.THRESH_OTSU);

            Mat fg = new Mat(image.size(), CvType.CV_8U);
            Imgproc.erode(binaryImage, fg, new Mat(), new Point(-1, -1), 0);

            Mat bg = new Mat(image.size(), CvType.CV_8U);
            Imgproc.dilate(binaryImage, bg, new Mat(), new Point(-1, -1), 10);
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
            Mat source = Imgcodecs.imread("duc.jpg", Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            Mat destination = new Mat(source.rows(), source.cols(), source.type());

            destination = source;

            int erosion_size = 1;
            int dilation_size = 1;

            Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * erosion_size + 1, 2 * erosion_size + 1));
            Imgproc.erode(source, destination, element);
            Imgcodecs.imwrite("BinaryImageErosion.jpg", destination);

            destination = source;

            Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * dilation_size + 1, 2 * dilation_size + 1));
            Imgproc.dilate(source, destination, element1);
            Imgcodecs.imwrite("BinaryImageDilation.jpg", destination);
            LoadImg("BinaryImageErosion.jpg");
            LoadImg("BinaryImageDilation.jpg");
        } catch (Exception e) {
            System.out.println("error: " + e.getMessage());
        }
    }

    public void LoadImg(String src) {
        Mat m = Imgcodecs.imread(src, Imgcodecs.CV_LOAD_IMAGE_COLOR);
        LoadImage loadImage = new LoadImage(src, m);
    }

}

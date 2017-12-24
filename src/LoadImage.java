

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import java.awt.*;
import javax.swing.*;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class LoadImage extends JFrame {

    public LoadImage(String imgStr, Mat m) {
        Imgcodecs.imwrite(imgStr, m);
        JFrame frame = new JFrame("My GUI");

        frame.setResizable(true);
        frame.setLocationRelativeTo(null);

// Inserts the image icon
        ImageIcon image = new ImageIcon(imgStr);
        frame.setSize(image.getIconWidth() + 10, image.getIconHeight() + 35);
// Draw the Image data into the BufferedImage
        JLabel label1 = new JLabel(" ", image, JLabel.CENTER);
        frame.getContentPane().add(label1);

        frame.validate();
        frame.setVisible(true);
    }

}

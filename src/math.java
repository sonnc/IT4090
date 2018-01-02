
import java.util.Comparator;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author sonng
 */
public class math {

    private int stt;    
    private int x;
    private int y;
    private String dai;
    private int rong;

    public math(int stt, int x, int y, String dai, int rong) {
        this.stt = stt;
        this.dai = dai;
        this.rong = rong;
        this.x = x;
        this.y = y;
    }

    public int getStt() {
        return stt;
    }

    public void setStt(int stt) {
        this.stt = stt;
    }

    public int getX() {
        return x;
    }

    public void setX(int x) {
        this.x = x;
    }

    public int getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }

    public String getDai() {
        return dai;
    }

    public void setDai(String dai) {
        this.dai = dai;
    }

    public int getRong() {
        return rong;
    }

    public void setRong(int rong) {
        this.rong = rong;
    }
    
    @Override
    public String toString() {
        return stt + " - " + x + " - " + y + " - " + dai + " - " + rong;
    }
    // sắp xếp sinh viên theo tên và theo khoa.

    public static Comparator<math> compare = new Comparator<math>() {

        @Override
        public int compare(math o1, math o2) {
            return o1.dai.compareTo(o2.dai);
        }
    };
   

}

package pers.season.vml.util;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImUtils {

	public static void imshow(Mat img) {
		imshow(img, 1f);
	}

	public static void imshow(Mat img, float scale) {
		imshow(new JFrame(), img, scale);
	}

	public static void imshow(JFrame win, Mat img, float scale) {
		win.getContentPane().removeAll();
		win.getContentPane().setLayout(null);
		JLabel lbl = new JLabel("");
		lbl.setBounds(0, 0, (int) (img.width() * scale), (int) (img.height() * scale));
		lbl.setIcon(new ImageIcon(ImUtils.encodeImage(img, scale)));
		win.getContentPane().add(lbl);
		win.setSize((int) (img.width() * scale) + 20, (int) (img.height() * scale) + 40);
		win.setVisible(true);
		win.repaint();
	}

	public static BufferedImage encodeImage(Mat image, float scale) {
		Size size = image.size();
		size.height *= scale;
		size.width *= scale;
		return encodeImage(image, size);
	}

	public static BufferedImage encodeImage(Mat image, Size resize) {
		Mat tImg = image.clone();
		if (resize != null)
			Imgproc.resize(tImg, tImg, resize);

		MatOfByte matOfByte = new MatOfByte();
		Imgcodecs.imencode(".bmp", tImg, matOfByte);

		byte[] byteArray = matOfByte.toArray();

		BufferedImage bufImage = null;
		try {
			InputStream in = new ByteArrayInputStream(byteArray);
			bufImage = ImageIO.read(in);
			return bufImage;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	public static Mat loadMat(String file) {
		try {
			BufferedReader in = new BufferedReader(new FileReader(file));

			Mat result = new Mat();
			String line;
			while ((line = in.readLine()) != null) {
				String[] lineseq = line.split(",");
				Mat result_line = new Mat(1, lineseq.length, CvType.CV_32F);
				float[] data = new float[lineseq.length];
				for (int i = 0; i < data.length; i++)
					data[i] = Float.parseFloat(lineseq[i]);
				result_line.put(0, 0, data);
				result.push_back(result_line);
			}
			in.close();
			System.gc();
			return result;

		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	public static void saveMatAsInt(Mat mat, String file) {
		try {
			new File(file).getParentFile().mkdirs();
			BufferedWriter out = new BufferedWriter(new FileWriter(file));

			// out.write(mat.rows() + "," + mat.cols() + "," + mat.type()+"\n");

			for (int i = 0; i < mat.rows(); i++) {
				float[] data = new float[mat.cols() * mat.channels()];
				mat.get(i, 0, data);

				for (int ii = 0; ii < data.length; ii++) {
					if (ii != data.length - 1)
						out.write(Math.round(data[ii]) + ",");
					else
						out.write(Math.round(data[ii]) + "\n");
				}

			}

			out.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void saveMat(Mat mat, String file) {
		try {
			new File(file).getParentFile().mkdirs();
			BufferedWriter out = new BufferedWriter(new FileWriter(file));

			// out.write(mat.rows() + "," + mat.cols() + "," + mat.type()+"\n");

			for (int i = 0; i < mat.rows(); i++) {
				float[] data = new float[mat.cols() * mat.channels()];
				mat.get(i, 0, data);

				for (int ii = 0; ii < data.length; ii++) {
					if (ii != data.length - 1)
						out.write(data[ii] + ",");
					else
						out.write(data[ii] + "\n");
				}

			}

			out.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public static void sharpen(Mat src, Mat dst) {
		Mat kernel = new Mat(3, 3, CvType.CV_32F);
		kernel.put(0, 0, -1, -1, -1, -1, 9, -1, -1, -1, -1);
		Imgproc.filter2D(src, dst, -1, kernel);

	}

	public static void showDelaunay(Mat pts, int[][] delaunay, int width, int height) {
		showDelaunay(new JFrame(), pts, delaunay, width, height);
	}

	public static void showDelaunay(JFrame win, Mat pts, int[][] delaunay, int width, int height) {
		Mat delaunay_show = Mat.zeros(height, width, CvType.CV_32F);
		for (int i = 0; i < delaunay.length; i++) {
			int p1c = delaunay[i][0];
			int p2c = delaunay[i][1];
			int p3c = delaunay[i][2];
			double p1x = pts.get(p1c * 2, 0)[0];
			double p2x = pts.get(p2c * 2, 0)[0];
			double p3x = pts.get(p3c * 2, 0)[0];
			double p1y = pts.get(p1c * 2 + 1, 0)[0];
			double p2y = pts.get(p2c * 2 + 1, 0)[0];
			double p3y = pts.get(p3c * 2 + 1, 0)[0];
			Imgproc.line(delaunay_show, new Point(p1x, p1y), new Point(p2x, p2y), new Scalar(255));
			Imgproc.line(delaunay_show, new Point(p3x, p3y), new Point(p2x, p2y), new Scalar(255));
			Imgproc.line(delaunay_show, new Point(p1x, p1y), new Point(p3x, p3y), new Scalar(255));
		}
		ImUtils.imshow(win, delaunay_show, 1);
	}

	private static long timingRecord = 0;

	public static void startTiming() {
		timingRecord = System.nanoTime();
	}

	public static double getTiming() {
		return (System.nanoTime() - timingRecord) / 1000000.0;
	}

	public static void printTiming() {
		System.out.println("timing : " + getTiming());
	}

	public static double getCostE(Mat cost) {
		Mat result = new Mat();
		Core.gemm(cost.t(), cost, 1, new Mat(), 0, result);
		return result.get(0, 0)[0];
	}
}

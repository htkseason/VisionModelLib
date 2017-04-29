package pers.season.vml.statistics.shape;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public class ShapeModel {

	public Mat V, e;

	public int X_SIZE, Z_SIZE;

	protected double transPerPixel, scalePerPixel;

	public static ShapeModel load(String dataPath, String V_name, String e_name) {
		ShapeModel sm = new ShapeModel();
		sm.V = ImUtils.loadMat(dataPath + V_name);
		sm.e = ImUtils.loadMat(dataPath + e_name);
		sm.X_SIZE = sm.V.rows();
		sm.Z_SIZE = sm.V.cols();
		sm.transPerPixel = sm.calcTransPerPixel();
		sm.scalePerPixel = sm.calcScalePerPixel();
		System.out.println("ShapeModel inited. " + sm.X_SIZE + " --> " + sm.Z_SIZE);
		return sm;
	}

	public void clamp(Mat Z, double maxBias) {
		double scale = getScale(Z);
		for (int i = 4; i < Z_SIZE; i++) {
			double p = Z.get(i, 0)[0];
			if (Math.abs(p) > scale * e.get(i, 0)[0] * maxBias)
				Z.put(i, 0, Math.signum(p) * scale * e.get(i, 0)[0] * maxBias);
		}
	}

	public void printTo(Mat Z, Mat dst) {
		Mat X = getXfromZ(Z);
		for (int i = 0; i < X.rows() / 2; i++) {
			Imgproc.circle(dst, new Point(X.get(i * 2, 0)[0], X.get(i * 2 + 1, 0)[0]), 2, new Scalar(255));
		}
	}

	public Mat getXfromZ(Mat Z) {
		Mat result = new Mat();
		Core.gemm(V, Z, 1, new Mat(), 0, result);
		return result;
	}

	public double getScale(Mat Z) {
		double a = Z.get(0, 0)[0];
		double b = Z.get(1, 0)[0];
		return Math.sqrt(a * a + b * b);
	}

	public Mat getZfromX(Mat X) {
		Mat result = new Mat();
		Core.gemm(V.t(), X, 1, new Mat(), 0, result);
		return result;
	}

	public Mat getZe4fromZ(Mat Z) {
		Mat result = Mat.zeros(Z.rows() - 4, 1, CvType.CV_32F);
		double a = Z.get(0, 0)[0];
		double b = Z.get(1, 0)[0];
		double scale = Math.sqrt(a * a + b * b);

		Core.divide(Z.rowRange(4, Z.rows()), new Scalar(scale), result);

		return result;
	}

	public Mat getZe4fromX(Mat X) {
		return getZe4fromZ(getZfromX(X));
	}

	public double getTransPerPixel() {
		return transPerPixel;
	}

	public double getScalePerPixel() {
		return scalePerPixel;
	}
	
	public Rect getLocation(Mat Z) {
		Mat X = getXfromZ(Z);
		double size = getScale(Z) / getScalePerPixel();
		double mx = 0, my = 0;
		for (int i = 0; i < X.rows() / 2; i++) {
			mx += X.get(i * 2, 0)[0];
			my += X.get(i * 2 + 1, 0)[0];
		}
		mx /= X.rows() / 2;
		my /= X.rows() / 2;
		return new Rect((int) (mx - size / 2), (int) (my - size / 2), (int) size, (int) size);
	}

	protected double calcTransPerPixel() {
		Mat Z = Mat.zeros(Z_SIZE, 1, CvType.CV_32F);
		Z.put(0, 0, 1);
		Mat X = getXfromZ(Z);
		for (int i = 0; i < X.rows() / 2; i++) {
			X.put(i * 2, 0, X.get(i * 2, 0)[0] + 10);
			X.put(i * 2 + 1, 0, X.get(i * 2 + 1, 0)[0] + 10);
		}
		Z = getZfromX(X);
		return (Z.get(3, 0)[0] + Z.get(2, 0)[0]) / 2 / 10;
	}

	protected double calcScalePerPixel() {
		Mat Z = Mat.zeros(Z_SIZE, 1, CvType.CV_32F);
		Z.put(0, 0, 1);
		Mat X = getXfromZ(Z);
		Core.normalize(X, X, -10 / 2, 10 / 2, Core.NORM_MINMAX);
		Z = getZfromX(X);
		return Z.get(0, 0)[0] / 10;
	}
}

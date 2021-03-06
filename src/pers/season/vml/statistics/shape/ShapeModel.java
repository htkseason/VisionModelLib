package pers.season.vml.statistics.shape;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public class ShapeModel {

	public Mat V, e;

	public int X_SIZE, Z_SIZE;

	protected double transPerPixel, scalePerPixel;

	protected ShapeModel() {

	}

	public static ShapeModel load(String dataPath, String V_name, String e_name) {
		return load(ImUtils.loadMat(dataPath + V_name), ImUtils.loadMat(dataPath + e_name));
	}

	public static ShapeModel load(Mat V, Mat e) {
		ShapeModel sm = new ShapeModel();
		sm.V = V.clone();
		sm.e = e.clone();
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

	public void setScale(Mat Z, double scale) {
		double quotient = scale / getScale(Z);
		Z.put(0, 0, Z.get(0, 0)[0] * quotient);
		Z.put(1, 0, Z.get(1, 0)[0] * quotient);
	}

	public double getRadian(Mat Z) {
		double scale = getScale(Z);
		double cos = Z.get(0, 0)[0] / scale;
		double sin = Z.get(1, 0)[0] / scale;
		if (sin == 0)
			return cos > 0 ? Math.PI * 0.0 : Math.PI * 1.0;
		if (cos == 0)
			return sin > 0 ? Math.PI * 0.5 : Math.PI * 1.5;
		double radian = Math.atan(sin / cos);
		return cos > 0 ? radian : sin > 0 ? Math.PI + radian : -Math.PI + radian;
	}

	public void setRadian(Mat Z, double radian) {
		double scale = getScale(Z);
		Z.put(0, 0, scale * Math.cos(radian));
		Z.put(1, 0, scale * Math.sin(radian));
	}

	public Point getOffset(Mat Z) {
		return new Point(Z.get(2, 0)[0] / transPerPixel, Z.get(3, 0)[0] / transPerPixel);
	}

	public void setOffset(Mat Z, Point offset) {
		Z.put(2, 0, offset.x * transPerPixel);
		Z.put(3, 0, offset.y * transPerPixel);
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

	public RotatedRect getLocation(Mat Z) {
		Mat X = getXfromZ(Z);
		double size = getScale(Z) / getScalePerPixel();
		double radian = getRadian(Z);
		double mx = 0, my = 0;
		for (int i = 0; i < X.rows() / 2; i++) {
			mx += X.get(i * 2, 0)[0];
			my += X.get(i * 2 + 1, 0)[0];
		}
		mx /= X.rows() / 2;
		my /= X.rows() / 2;
		double degree = 180 * radian / Math.PI;
		return new RotatedRect(new Point(mx, my), new Size(size, size), degree);
	}

	public Rect getLocationRect(Mat Z) {
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

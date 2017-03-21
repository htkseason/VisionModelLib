package priv.season.aam.statistics.shape;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import priv.season.aam.util.MuctData;

import priv.season.aam.util.ImUtils;

public class ShapeModel {

	public static Mat V, e;

	public static int X_SIZE, Z_SIZE;

	public static void init(String dataPath, String V_name, String e_name) {
		V = ImUtils.loadMat(dataPath + V_name);
		e = ImUtils.loadMat(dataPath + e_name);
		X_SIZE = V.rows();
		Z_SIZE = V.cols();

	}

	public static void clamp(Mat Z, double maxBias) {
		for (int i = 4; i < Z_SIZE; i++) {
			double p = Z.get(i, 0)[0];
			if (Math.abs(p) > e.get(i, 0)[0] * maxBias)
				Z.put(i, 0, Math.signum(p) * e.get(i, 0)[0] * maxBias);
		}
	}

	public static void printTo(Mat dst, Mat X) {
		for (int i = 0; i < X.rows() / 2; i++) {
			Imgproc.circle(dst, new Point(X.get(i * 2, 0)[0], X.get(i * 2 + 1, 0)[0]), 2, new Scalar(255));
		}
	}

	public static Mat getXfromZ(Mat Z) {
		Mat result = new Mat();
		Core.gemm(V, Z, 1, new Mat(), 0, result);
		return result;
	}

	public static double getScale(Mat Z) {
		double a = Z.get(0, 0)[0];
		double b = Z.get(1, 0)[0];
		return Math.sqrt(a * a + b * b);
	}

	public static Mat getZfromX(Mat X) {
		Mat result = new Mat();
		Core.gemm(V.t(), X, 1, new Mat(), 0, result);
		return result;
	}

	public static Mat getZe4fromZ(Mat Z) {
		Mat result = Mat.zeros(Z.rows() - 4, 1, CvType.CV_64F);
		double a = Z.get(0, 0)[0];
		double b = Z.get(1, 0)[0];
		double scale = Math.sqrt(a * a + b * b);

		Core.divide(Z.rowRange(4, Z.rows()), new Scalar(scale), result);

		return result;
	}

	public static Mat getZe4fromX(Mat X) {
		return getZe4fromZ(getZfromX(X));
	}

}

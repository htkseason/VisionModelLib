package pers.season.vml.statistics.shape;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class ShapeInstance extends ShapeModel {
	public Mat Z;

	public ShapeInstance(Mat pts) {
		Z = ShapeModel.getZfromX(pts);
	}

	public ShapeInstance(double scale, double theta, double transx, double transy) {
		Z = Mat.zeros(ShapeModel.Z_SIZE, 1, CvType.CV_32F);
		Z.put(0, 0, 1);
		Mat X = ShapeModel.getXfromZ(Z);
		Core.normalize(X, X, -scale / 2, scale / 2, Core.NORM_MINMAX);
		for (int i = 0; i < X.rows() / 2; i++) {
			X.put(i * 2, 0, X.get(i * 2, 0)[0] + transx);
			X.put(i * 2 + 1, 0, X.get(i * 2 + 1, 0)[0] + transy);
		}
		Z = ShapeModel.getZfromX(X);
		Z.put(0, 0, Z.get(0, 0)[0] * Math.cos(theta));
		Z.put(1, 0, Z.get(0, 0)[0] * Math.sin(theta));
	}

	public void setFromZ(Mat Z) {
		this.Z = Z.clone();
	}

	public void setFromX(Mat X) {
		this.Z = ShapeModel.getZfromX(X);
	}

	public Mat getX() {
		return ShapeModel.getXfromZ(Z);
	}

	public Mat getZ() {
		return Z.clone();
	}

	public Mat getZe4() {
		return getZe4fromZ(Z);
	}

	public double getScale() {
		double a = Z.get(0, 0)[0];
		double b = Z.get(1, 0)[0];
		return Math.sqrt(a * a + b * b);
	}

	public void clamp(double maxBias) {
		clamp(Z, maxBias);
	}

	public void printTo(Mat dst) {
		printTo(Z, dst);
	}

}

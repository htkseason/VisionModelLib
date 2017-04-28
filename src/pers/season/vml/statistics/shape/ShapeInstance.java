package pers.season.vml.statistics.shape;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class ShapeInstance  {
	
	protected ShapeModel sm;
	public Mat Z;
	

	public ShapeInstance(ShapeModel sm) {
		this.sm = sm;
		Z = Mat.zeros(sm.Z_SIZE, 1, CvType.CV_32F);
	}
	
	public void setFromPts(Mat pts) {
		Z = sm.getZfromX(pts);
	}

	public void setFromParams(double scale, double theta, double transx, double transy) {
		Z = Mat.zeros(sm.Z_SIZE, 1, CvType.CV_32F);
		Z.put(0, 0, 1);
		Mat X = sm.getXfromZ(Z);
		Core.normalize(X, X, -scale / 2, scale / 2, Core.NORM_MINMAX);
		for (int i = 0; i < X.rows() / 2; i++) {
			X.put(i * 2, 0, X.get(i * 2, 0)[0] + transx);
			X.put(i * 2 + 1, 0, X.get(i * 2 + 1, 0)[0] + transy);
		}
		Z = sm.getZfromX(X);
		Z.put(0, 0, Z.get(0, 0)[0] * Math.cos(theta));
		Z.put(1, 0, Z.get(0, 0)[0] * Math.sin(theta));
	}

	
	
	public Mat getX() {
		return sm.getXfromZ(Z);
	}

	public Mat getZ() {
		return Z.clone();
	}

	public Mat getZe4() {
		return sm.getZe4fromZ(Z);
	}

	public double getScale() {
		double a = Z.get(0, 0)[0];
		double b = Z.get(1, 0)[0];
		return Math.sqrt(a * a + b * b);
	}

	public void clamp(double maxBias) {
		sm.clamp(Z, maxBias);
	}

	public void printTo(Mat dst) {
		sm.printTo(Z, dst);
	}
	
	public ShapeModel getShapeModel() {
		return sm;
	}

}

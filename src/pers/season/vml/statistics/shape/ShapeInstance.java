package pers.season.vml.statistics.shape;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;

public class ShapeInstance {

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

		// move to the mass center
		Mat mx = Mat.zeros(1, 1, CvType.CV_32F);
		Mat my = Mat.zeros(1, 1, CvType.CV_32F);
		for (int i = 0; i < X.rows() / 2; i++) {
			Core.add(mx, X.row(i * 2), mx);
			Core.add(my, X.row(i * 2 + 1), my);
		}
		Core.divide(mx, new Scalar(X.rows() / 2), mx);
		Core.divide(my, new Scalar(X.rows() / 2), my);
		for (int i = 0; i < X.rows() / 2; i++) {
			Core.subtract(X.row(i * 2), mx, X.row(i * 2));
			Core.subtract(X.row(i * 2 + 1), my, X.row(i * 2 + 1));
		}

		// apply offset
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
		return sm.getScale(Z);
	}

	public double getRadian() {
		return sm.getRadian(Z);
	}

	public void setScale(double scale) {
		sm.setScale(Z, scale);
	}

	public void setRadian(double radian) {
		sm.setRadian(Z, radian);
	}

	public Point getOffset() {
		return sm.getOffset(Z);
	}

	public void setOffset(Point offset) {
		sm.setOffset(Z, offset);
	}

	public RotatedRect getLocation() {
		return sm.getLocation(Z);
	}

	public Rect getLocationRect() {
		return sm.getLocationRect(Z);
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

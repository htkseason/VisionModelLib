package pers.season.vml.statistics.appearance;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import pers.season.vml.statistics.texture.TextureInstance;

public class AppearanceInstance {
	public Mat Z;
	protected AppearanceModel am;

	public AppearanceInstance(AppearanceModel am) {
		this.am = am;
		Z = Mat.zeros(am.Z_SIZE, 1, CvType.CV_32F);
	}

	public void setFromModels(Mat shapeZ, Mat textureZ) {
		this.Z = am.getZfromModel(shapeZ, textureZ);
	}

	public Mat getZ() {
		return Z.clone();
	}

	public Mat getX() {
		return am.getXfromZ(Z);
	}

	public Mat getShapeZ() {
		Mat X = am.getXfromZ(Z);
		double scale = am.sm.getScale(X);
		Core.multiply(X.rowRange(4, am.sm.Z_SIZE), new Scalar(scale), X.rowRange(4, am.sm.Z_SIZE));
		return X.rowRange(0, am.sm.Z_SIZE);
	}
	
	public Mat getTextureZ() {
		Mat X = am.getXfromZ(Z);
		return X.rowRange(am.sm.Z_SIZE, X.rows());
	}

	public void printTo(Mat dst, boolean showPts) {
		am.printTo(Z, dst, showPts);
	}

	public AppearanceModel getAppearanceModel() {
		return am;
	}
}

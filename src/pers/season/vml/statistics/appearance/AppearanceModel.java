package pers.season.vml.statistics.appearance;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.statistics.texture.TextureModel;
import pers.season.vml.util.ImUtils;

public class AppearanceModel {

	public Mat U, e;
	public double shapeWeight;

	public int Z_SIZE, X_SIZE;

	protected ShapeModel sm;
	protected TextureModel tm;

	public static AppearanceModel load(ShapeModel sm, TextureModel tm, String dataPath, String U_name, String e_name,
			String shapeWeight_name) {
		return load(sm, tm, ImUtils.loadMat(dataPath + U_name), ImUtils.loadMat(dataPath + e_name),
				ImUtils.loadMat(dataPath + shapeWeight_name));
	}

	public static AppearanceModel load(ShapeModel sm, TextureModel tm, Mat U, Mat e, Mat shapeWeight) {
		AppearanceModel am = new AppearanceModel();
		am.U = U.clone();
		am.e = e.clone();
		am.shapeWeight = shapeWeight.get(0, 0)[0];
		am.Z_SIZE = am.U.cols() + 4;
		am.X_SIZE = am.U.rows() + 4;
		am.sm = sm;
		am.tm = tm;

		System.out.println("AppearanceModel inited. " + am.X_SIZE + " --> " + am.Z_SIZE);
		return am;
	}

	// Z = trans(4) + app
	public Mat getZfromModel(Mat shapeZ, Mat textureZ) {
		Mat X = new Mat();
		X.push_back(sm.getZe4fromZ(shapeZ));
		X.push_back(textureZ);
		Core.multiply(X.rowRange(0, sm.Z_SIZE - 4), new Scalar(shapeWeight), X.rowRange(0, sm.Z_SIZE - 4));
		Mat Z = new Mat(Z_SIZE, 1, CvType.CV_32F);
		shapeZ.rowRange(0, 4).copyTo(Z.rowRange(0, 4));
		Core.gemm(U.t(), X, 1, new Mat(), 0, Z.rowRange(4, Z.rows()));
		return Z;
	}

	// X = trans(4) + shape_Ze4 + texture_Z
	public Mat getXfromZ(Mat Z) {
		Mat X = new Mat(X_SIZE, 1, CvType.CV_32F);
		Z.rowRange(0, 4).copyTo(X.rowRange(0, 4));
		Core.gemm(U, Z.rowRange(4, Z.rows()), 1, new Mat(), 0, X.rowRange(4, X.rows()));
		Core.divide(X.rowRange(4, sm.Z_SIZE), new Scalar(shapeWeight), X.rowRange(4, sm.Z_SIZE));
		return X;
	}

	public void printTo(Mat Z, Mat dst, boolean showPts) {
		Mat X = getXfromZ(Z);
		double scale = sm.getScale(X);
		Core.multiply(X.rowRange(4, sm.Z_SIZE), new Scalar(scale), X.rowRange(4, sm.Z_SIZE));
		Mat shapeZ = X.rowRange(0, sm.Z_SIZE);
		Mat textureZ = X.rowRange(sm.Z_SIZE, X.rows());
		tm.printTo(textureZ, dst, sm.getXfromZ(shapeZ));
		if (showPts)
			sm.printTo(shapeZ, dst);
	}

	public ShapeModel getShapeModel() {
		return sm;
	}

	public TextureModel getTextureModel() {
		return tm;
	}

}

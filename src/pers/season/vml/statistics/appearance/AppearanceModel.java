package pers.season.vml.statistics.appearance;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.statistics.texture.TextureModel;
import pers.season.vml.util.ImUtils;

public class AppearanceModel {

	public static Mat U, e;
	public static double shapeWeight;

	public static int Z_SIZE, X_SIZE;

	public static void init(String dataPath, String U_name, String e_name, String shapeWeight_name) {
		U = ImUtils.loadMat(dataPath + U_name);
		e = ImUtils.loadMat(dataPath + e_name);
		shapeWeight = ImUtils.loadMat(dataPath + shapeWeight_name).get(0, 0)[0];
		Z_SIZE = U.cols() + 4;
		X_SIZE = U.rows() + 4;

		System.out.println("AppearanceModel inited. " + X_SIZE + " --> " + Z_SIZE);
	}

	public static void clamp(Mat Z, double maxBias) {
		for (int i = 4; i < Z_SIZE; i++) {
			double p = Z.get(i, 0)[0];
			if (Math.abs(p) > e.get(i - 4, 0)[0] * maxBias)
				Z.put(i, 0, Math.signum(p) * e.get(i - 4, 0)[0] * maxBias);
		}
	}

	// Z = trans(4) + app
	public static Mat getZfromX(Mat shapeZ, Mat textureZ) {
		Mat Z = new Mat();
		Z.push_back(shapeZ.rowRange(0, 4));
		Z.push_back(ShapeModel.getZe4fromZ(shapeZ));
		Z.push_back(textureZ);
		Mat Z_non_rigid = Z.rowRange(4, Z.rows());
		Mat Z_shape_e4 = Z.rowRange(4, ShapeModel.Z_SIZE);
		Core.multiply(Z_shape_e4, new Scalar(shapeWeight), Z_shape_e4);
		Core.gemm(U.t(), Z_non_rigid, 1, new Mat(), 0, Z_non_rigid);
		return Z.rowRange(0, Z_SIZE);
	}

	// X = trans(4) + shape_Xe4 + texture_X
	public static Mat getXfromZ(Mat Z) {
		Mat X = new Mat(X_SIZE, 1, CvType.CV_32F);
		Z.rowRange(0, 4).copyTo(X.rowRange(0, 4));
		Mat X_non_rigid = X.rowRange(4, X.rows());
		Mat X_shape_e4 = X.rowRange(4, ShapeModel.Z_SIZE);
		Core.gemm(U, Z.rowRange(4, Z.rows()), 1, new Mat(), 0, X_non_rigid);
		Core.divide(X_shape_e4, new Scalar(shapeWeight), X_shape_e4);
		return X;
	}

	public static void printTo(Mat Z, Mat dst) {
		Mat X = getXfromZ(Z);
		double scale = ShapeModel.getScale(X);
		Core.multiply(X.rowRange(4, ShapeModel.Z_SIZE), new Scalar(scale), X.rowRange(4, ShapeModel.Z_SIZE));
		Mat shapeZ = X.rowRange(0, ShapeModel.Z_SIZE);
		Mat textureZ = X.rowRange(ShapeModel.Z_SIZE, X.rows());
		TextureModel.printTo(textureZ, dst, ShapeModel.getXfromZ(shapeZ));
		// ShapeModel.printTo(dst, ShapeModel.getXfromZ(shapeZ));
	}

}

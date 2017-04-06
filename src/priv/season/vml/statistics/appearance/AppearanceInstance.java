package priv.season.vml.statistics.appearance;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import priv.season.vml.statistics.shape.ShapeModel;
import priv.season.vml.statistics.texture.TextureModel;

public class AppearanceInstance extends AppearanceModel {
	public Mat Z;


	
	public AppearanceInstance(Mat shapeZ, Mat textureZ) {
		this.Z = getZfromX(shapeZ, textureZ);
	}
	


	public Mat getZ() {
		return Z;
	}



	public void clamp(double maxBias) {
		clamp(Z, maxBias);
	}



	public void printTo(Mat dst) {
		Mat X = getXfromZ(Z);
		double scale = ShapeModel.getScale(X);
		Core.multiply(X.rowRange(4, ShapeModel.Z_SIZE), new Scalar(scale), X.rowRange(4, ShapeModel.Z_SIZE));
		Mat shapeZ = X.rowRange(0, ShapeModel.Z_SIZE);
		Mat textureZ = X.rowRange(ShapeModel.Z_SIZE, X.rows());
		TextureModel.printTo(dst, TextureModel.getXfromZ(textureZ), ShapeModel.getXfromZ(shapeZ));
		// ShapeModel.printTo(dst, ShapeModel.getXfromZ(shapeZ));
	}
}

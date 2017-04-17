package pers.season.vml.statistics.appearance;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.statistics.texture.TextureModel;

public class AppearanceInstance extends AppearanceModel {
	public Mat Z;

	public AppearanceInstance(Mat shapeZ, Mat textureZ) {
		this.Z = getZfromX(shapeZ, textureZ);
	}

	public Mat getZ() {
		return Z.clone();
	}

	public void clamp(double maxBias) {
		clamp(Z, maxBias);
	}

	public void printTo(Mat dst) {
		AppearanceModel.printTo(Z, dst);
	}
}

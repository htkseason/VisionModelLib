package pers.season.vml.statistics.appearance;

import org.opencv.core.Mat;

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

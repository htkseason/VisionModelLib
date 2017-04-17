package pers.season.vml.statistics.texture;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class TextureInstance extends TextureModel {
	public Mat Z;

	public TextureInstance() {
		Z = Mat.zeros(TextureModel.Z_SIZE, 1, CvType.CV_32F);
	}

	public TextureInstance(Mat pic, Mat pts) {
		Z = TextureModel.getZfromX(TextureModel.getNormFace(pic, pts));
	}

	public Mat getZ() {
		return Z.clone();
	}

	public Mat getX() {
		return TextureModel.getXfromZ(Z);
	}

	public void printTo(Mat dst) {
		printTo(dst, TextureModel.stdShape);
	}
	
	public void printTo(Mat dst, Mat shape) {
		printTo(Z, dst, shape);
	}



	public void clamp(double maxBias) {
		clamp(Z, maxBias);
	}
}

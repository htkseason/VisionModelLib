package priv.season.aam.statistics.texture;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class TextureInstance extends TextureModel {
	public Mat Z;

	public TextureInstance() {
		Z = Mat.zeros(TextureModel.Z_SIZE, 1, CvType.CV_64F);
	}

	public TextureInstance(Mat pic, Mat pts) {
		Z = TextureModel.getZfromX(TextureModel.getNormFace(pic, pts));
	}

	public Mat getZ() {
		return Z;
	}

	public Mat getX() {
		return TextureModel.getXfromZ(Z);
	}

	public void printTo(Mat dst) {
		printTo(dst, getX(), stdShape);
	}

	public void printTo(Mat dst, Mat pts) {
		printTo(dst, getX(), pts);
	}

	public void clamp(double maxBias) {
		clamp(Z, maxBias);
	}
}

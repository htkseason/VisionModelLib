package pers.season.vml.statistics.texture;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class TextureInstance {
	
	protected TextureModel tm;
	public Mat Z;
	
	public TextureInstance(TextureModel tm) {
		this.tm = tm;
		Z = Mat.zeros(tm.Z_SIZE, 1, CvType.CV_32F);
	}

	public void setFromPic(Mat pic, Mat pts) {
		Z = tm.getZfromX(tm.getNormFace(pic, pts));
	}

	public Mat getZ() {
		return Z.clone();
	}

	public Mat getX() {
		return tm.getXfromZ(Z);
	}

	public void printTo(Mat dst) {
		this.printTo(dst, tm.stdShape);
	}
	
	public void printTo(Mat dst, Mat shape) {
		tm.printTo(Z, dst, shape);
	}
	
	public TextureModel getTextureModel() {
		return tm;
	}



}

package pers.season.vml.ar;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class CameraData {

	public final static Mat MY_CAMERA = new Mat(3, 3, CvType.CV_64F);

	static {
		MY_CAMERA.put(0, 0, new double[] { 810, 0, 325, 0, 810, 260, 0, 0, 1 });
	}
}

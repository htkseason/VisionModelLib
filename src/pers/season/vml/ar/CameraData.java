package pers.season.vml.ar;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class CameraData {

	public final static Mat MY_CAMERA = new Mat(3, 3, CvType.CV_64F);
	public final static Mat MY_CAMERA_DISTORTION = new Mat(5, 1, CvType.CV_64F);

	static {
		MY_CAMERA.put(0, 0, new double[] { 810, 0, 325, 0, 810, 260, 0, 0, 1 });
		MY_CAMERA_DISTORTION.put(0, 0, new double[] { -0.0525468933514149	,0.221184896836067,
				-0.00255453088299326 ,-0.00100957297737812,-0.528571608523801});

	}
}

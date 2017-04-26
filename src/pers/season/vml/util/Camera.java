package pers.season.vml.util;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;



public class Camera {
	protected static VideoCapture vc;
	static {
		vc = new VideoCapture(0);
	}
	
	public static Mat capture() {
		Mat ret = new Mat() ;
		vc.read(ret);
		return ret;
	}
}

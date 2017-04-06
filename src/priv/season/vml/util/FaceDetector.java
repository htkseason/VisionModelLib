package priv.season.vml.util;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.objdetect.CascadeClassifier;


public class FaceDetector {
	private static CascadeClassifier faceDetector;
	static {
		faceDetector = new CascadeClassifier("lbpcascade_frontalface.xml");
	}
	public static Rect searchFace(Mat img) {
		MatOfRect faceResult = new MatOfRect();
		faceDetector.detectMultiScale(img, faceResult);
		
		List<Rect> faceRects = Arrays.asList(faceResult.toArray());
		if (faceRects.size() < 1) {
			return null;
		}
		// sort by size
		faceRects.sort(new Comparator<Rect>() {
			@Override
			public int compare(Rect o1, Rect o2) {
				return -Double.compare(o1.area(), o2.area());
			}
		});
		return faceRects.get(0);
	}
}

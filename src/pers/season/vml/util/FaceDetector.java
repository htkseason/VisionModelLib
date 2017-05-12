package pers.season.vml.util;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.objdetect.CascadeClassifier;

public class FaceDetector {
	protected CascadeClassifier faceDetector;

	protected FaceDetector() {
		
	}
	public static FaceDetector load(String file) {
		FaceDetector fd = new FaceDetector();
		fd.faceDetector = new CascadeClassifier(file);
		return fd;
	}

	public Rect[] searchFace(Mat img) {
		MatOfRect faceResult = new MatOfRect();
		faceDetector.detectMultiScale(img, faceResult);

		Rect[] faceRects = faceResult.toArray();
		Arrays.sort(faceRects,  new Comparator<Rect>() {
			@Override
			public int compare(Rect o1, Rect o2) {
				return -Double.compare(o1.area(), o2.area());
			}
		}) ;

		return faceRects;
	}
}

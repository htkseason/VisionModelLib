package pers.season.vml.ar;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.util.ImUtils;

public class MarkerDetector {

	List<Mat> homoList = new ArrayList<Mat>(); 
	public MarkerDetector() {
		
	}
	
	public void findMarkers(Mat pic) {

		//Imgproc.adaptiveThreshold(pic, pic, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 7, 7);
		Imgproc.threshold(pic, pic, 0, 255, Imgproc.THRESH_OTSU);
		ImUtils.imshow(pic);
		List<MatOfPoint> countours = new ArrayList<MatOfPoint>();
		
		Imgproc.findContours(pic, countours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
		
		
		pic = Mat.zeros(pic.size(), pic.type());
		for (int i=0;i<countours.size();i++) {
			if (countours.get(i).rows()< 50)
				continue;
			MatOfPoint2f curve = new MatOfPoint2f();
			MatOfPoint2f countour = new MatOfPoint2f();
			countours.get(i).convertTo(countour, CvType.CV_32F);
			double eps = countours.get(i).rows() * 0.05;
			Imgproc.approxPolyDP(countour, curve, eps, true);
			
			if (curve.rows() != 4)
				continue;
			Imgproc.drawContours(pic, countours, i, new Scalar(255), 1);
			System.out.println(curve.size());
		}
		ImUtils.imshow(pic);
	}
	
}

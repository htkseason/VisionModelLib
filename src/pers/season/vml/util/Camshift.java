package pers.season.vml.util;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

public class Camshift {
	TermCriteria tc= new TermCriteria(TermCriteria.EPS | TermCriteria.MAX_ITER, 10, 1);
	MatOfFloat hrange = new MatOfFloat(new float[] { 0, 180 });
	MatOfInt channels = new MatOfInt(new int[] { 0 });
	MatOfInt hSize = new MatOfInt(new int[] { 18 });
	
	Mat hue, hist, mask = new Mat();
	Mat backProject = new Mat();

	public Scalar lowerb = new Scalar(0,0,0);
	public Scalar upperb = new Scalar(0,0,0);
	

	public Rect trackWindow = new Rect();

	public Camshift() {
	}
	
	public Mat getBackProject() {
	
		return backProject;
	}

	public void initHist(Mat hsv, Rect roi) {
		AFT(hsv,roi);
		
		Core.inRange(hsv, lowerb, upperb, mask);
		List<Mat> hsvChLst = new ArrayList<Mat>();
		Core.split(hsv, hsvChLst);
		hue = hsvChLst.get(0);
		
		Mat imgroi = hue.submat(roi);
		Mat maskroi = mask.submat(roi);
		
		List<Mat> images = new ArrayList<Mat>();
		images.add(imgroi);
		hist = new Mat();
		Imgproc.calcHist(images, channels, maskroi, hist, hSize, hrange);
		
		Core.normalize(hist, hist, 0, 255, Core.NORM_MINMAX);
		
		trackWindow = roi.clone();
		
		
	}

	public RotatedRect updata(Mat hsv) {
		Core.inRange(hsv, lowerb, upperb, mask);
		List<Mat> hsvChLst = new ArrayList<Mat>();
		Core.split(hsv, hsvChLst);
		hue = hsvChLst.get(0);
		
		
		List<Mat> images = new ArrayList<Mat>();
		images.add(hue);

		Imgproc.calcBackProject(images, channels, hist, backProject, hrange, 1);
		Core.bitwise_and(backProject, mask, backProject);
		try {
			RotatedRect result = Video.CamShift(backProject, trackWindow, tc);
			return result;
		} catch(Exception e) {
			return null;
		}
	}
	
	

	
	//Adjust hsv filter mask algorithm (AFT) Auto Fitted Threshold
	
	public void AFT(Mat hsv, Rect roi) {
		int bestlv = 0;
		int besthv = 255;
		int bestls = 0;
		int besths = 255;
		{
		int lv=bestlv;
		double bestPoint = 0;
		compareArea = -1;
		while (lv++<250) {
			Scalar lb= new Scalar(0, 10, lv);
			Scalar ub= new Scalar(180, 255, 255);
			Core.inRange(hsv, lb, ub, mask);
			double point = AftPoint(mask, roi);
			if (point > bestPoint) {
				bestPoint = point;
				bestlv = lv;
			}
		}
		System.out.println("AFT finds lv " + bestlv + " scores " + bestPoint);
		}
		{
		int hv=besthv;
		double bestPoint = 0;
		compareArea = -1;
		while (hv-->bestlv) {
			Scalar lb= new Scalar(0, 0, bestlv);
			Scalar ub= new Scalar(180, 255, hv);
			Core.inRange(hsv, lb, ub, mask);
			double point = AftPoint(mask, roi);
			if (point > bestPoint) {
				bestPoint = point;
				besthv = hv;
			}
		}
		System.out.println("AFT finds hv " + besthv + " scores " + bestPoint);
		}
		/*
		{
		int ls=bestls;		
		double bestPoint = 0;
		compareArea = -1;
		while (ls++<250) {
			Scalar lb= new Scalar(0, ls, 0);
			Scalar ub= new Scalar(180, 255, 255);
			Core.inRange(hsv, lb, ub, mask);
			double point = AftPoint(mask, roi);
			if (point > bestPoint) {
				bestPoint = point;
				bestls = ls;
			}
		}
		System.out.println("AFT finds ls " + bestls + " scores " + bestPoint);
		}
		{
		int hs=besths;
		double bestPoint = 0;
		compareArea = -1;
		while (hs-->bestls) {
			Scalar lb= new Scalar(0, 0, 0);
			Scalar ub= new Scalar(180, hs, 255);
			Core.inRange(hsv, lb, ub, mask);
			double point = AftPoint(mask, roi);
			if (point > bestPoint) {
				bestPoint = point;
				besths = hs;
			}
		}
		System.out.println("AFT finds hs " + besths + " scores " + bestPoint);
		}*/
		lowerb.set(new double[] {0,10,bestlv});
		upperb.set(new double[] {180,255,besthv});
	}
	
	double compareArea = -1;
	public double AftPoint(Mat bp, Rect roi) {
		Mat ellmask = Mat.zeros(roi.size(),CvType.CV_8U);
		Imgproc.ellipse(ellmask, new Point(ellmask.width()/2,  ellmask.height()/2),new Size(ellmask.width()/2, ellmask.height()/2), 0, 0, 360, new Scalar(1), -1, 8,0);
		
		double s2 = Core.sumElems(bp.submat(roi).mul(ellmask)).val[0]; //roi sum
		double s1 = Core.sumElems(bp).val[0] - s2;  //other sum
		
		if (compareArea == -1) 
			compareArea = s1/255;
		
		double v1 = 1 - s1/(compareArea*255);
		double v2 = s2/(roi.height*roi.width*255);
		v1=v1<0?v1=0:v1;
		
		double result = 2.0*v2*v1/(v1+v2);
		return result;

	}

}

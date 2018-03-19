package pers.season.vml.ar;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

public class ArUcoDetector {
	public double perimeterThreshold;
	public double areaThreshold;
	public int refineWinSizeDivisor;

	public ArUcoDetector(double perimeterThreshold, double areaThreshold, int refineWinSizeDivisor) {
		this.perimeterThreshold = perimeterThreshold;
		this.areaThreshold = areaThreshold;
		this.refineWinSizeDivisor = refineWinSizeDivisor;
	}

	public List<MatOfPoint2f> findMarkers(Mat pic) {
		Mat otsuPic = new Mat();
		List<MatOfPoint2f> ret = new ArrayList<MatOfPoint2f>();
		Imgproc.threshold(pic, otsuPic, 0, 255, Imgproc.THRESH_OTSU);

		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

		Imgproc.findContours(otsuPic, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
		for (int i = 0; i < contours.size(); i++) {
			if (contours.get(i).rows() < perimeterThreshold)
				continue;
			MatOfPoint2f contour = new MatOfPoint2f();
			MatOfPoint2f curve = new MatOfPoint2f();
			contours.get(i).convertTo(contour, CvType.CV_32F);

			Imgproc.approxPolyDP(contour, curve, contours.get(i).rows() * 0.05, true);
			if (curve.rows() != 4)
				continue;
			if (getArea(curve) < areaThreshold)
				continue;

			clockwiseMarker(curve);
			if (refineWinSizeDivisor > 0) {
				int winSize = contours.get(i).rows() / 4 / refineWinSizeDivisor;
				winSize = winSize < 2 ? 2 : winSize;
				Imgproc.cornerSubPix(pic, curve, new Size(winSize, winSize), new Size(-1, -1),
						new TermCriteria(TermCriteria.MAX_ITER, 30, 0.1));
			}

			ret.add(curve);
		}

		return ret;
	}

	protected void clockwiseMarker(Mat marker) {
		Mat v1 = new Mat(), v2 = new Mat();
		Core.subtract(marker.row(1), marker.row(0), v1);
		Core.subtract(marker.row(2), marker.row(0), v2);
		double clockwise = (v1.get(0, 0)[0] * v2.get(0, 0)[1]) - (v1.get(0, 0)[1] * v2.get(0, 0)[0]);
		if (clockwise < 0) {
			Core.bitwise_xor(marker.row(1), marker.row(3), marker.row(1));
			Core.bitwise_xor(marker.row(1), marker.row(3), marker.row(3));
			Core.bitwise_xor(marker.row(1), marker.row(3), marker.row(1));
		}
	}

	protected double getArea(Mat curve) {
		Mat dis = new Mat();
		double a1 = 0, b1 = 0, a2 = 0, b2 = 0, c = 0;
		for (int p = 0; p < 4; p++) {
			Core.subtract(curve.row(p), curve.row((p + 1) % 4), dis);
			if (p == 0)
				a1 = Math.sqrt(dis.dot(dis));
			if (p == 1)
				b1 = Math.sqrt(dis.dot(dis));
			if (p == 2)
				a2 = Math.sqrt(dis.dot(dis));
			if (p == 3)
				b2 = Math.sqrt(dis.dot(dis));
		}
		Core.subtract(curve.row(0), curve.row(2), dis);
		c = Math.sqrt(dis.dot(dis));
		return helen(a1, b1, c) + helen(a2, b2, c);

	}

	protected double helen(double a, double b, double c) {
		double p = (a + b + c) / 2.0;
		return Math.sqrt(p * (p - a) * (p - b) * (p - c));
	}

}

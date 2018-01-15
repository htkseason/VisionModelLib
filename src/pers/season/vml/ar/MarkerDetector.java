package pers.season.vml.ar;

import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.util.ImUtils;

public class MarkerDetector {

	public MarkerDetector() {

	}

	public List<MatOfPoint2f> findMarkers(Mat pic, int perimeterThreshold, double areaThreshold) {
		Mat otsuPic = new Mat();
		List<MatOfPoint2f> ret = new ArrayList<MatOfPoint2f>();
		Imgproc.threshold(pic, otsuPic, 0, 255, Imgproc.THRESH_OTSU);
		ImUtils.imshow(otsuPic);
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

			System.out.println("area = " + getArea(curve) + ", perimeter = " + contour.rows() + ", "
					+ parseMarker(pic, curve, new Size(7, 7), 0.1));
			ret.add(curve);
		}
		return ret;
	}

	public int parseMarker(Mat pic, MatOfPoint2f marker, Size blocks, double blockEpsilon) {
		int lengthPerBlock = 10;
		int width = (int) blocks.width * lengthPerBlock;
		int height = (int) blocks.height * lengthPerBlock;
		Mat srcPts = new Mat(4, 1, CvType.CV_32FC2);
		srcPts.put(0, 0, 0, 0, width, 0, width, height, 0, height);
		Mat homo = Imgproc.getPerspectiveTransform(srcPts, marker);

		Mat markerPic = new Mat();
		Imgproc.warpPerspective(pic, markerPic, homo, new Size(width, height), Imgproc.WARP_INVERSE_MAP);
		Imgproc.threshold(markerPic, markerPic, 0, 255, Imgproc.THRESH_OTSU);
		boolean[][] markerData = new boolean[(int) blocks.height][(int) blocks.width];
		for (int y = 0; y < blocks.height; y++)
			for (int x = 0; x < blocks.width; x++) {
				int nonZeroPixels = Core.countNonZero(markerPic.submat(y * lengthPerBlock, (y + 1) * lengthPerBlock,
						x * lengthPerBlock, (x + 1) * lengthPerBlock));
				if (Math.abs(0.5 - (double) nonZeroPixels / (lengthPerBlock * lengthPerBlock)) < blockEpsilon)
					return -1;
				markerData[y][x] = nonZeroPixels > lengthPerBlock * lengthPerBlock / 2 ? true : false;
			}

		// check validation
		for (int y = 0; y < blocks.height; y++)
			for (int x = 0; x < blocks.width; x++) {
				if (x == 0 || x == blocks.width || y == 0 || y == blocks.height)
					if (markerData[y][x] != false)
						return -1;
				
			}
		
		

		ImUtils.imshow(markerPic);
		return 1;
	}

	private void clockwiseMarker(Mat marker) {
		Mat v1 = new Mat(), v2 = new Mat();
		Core.subtract(marker.row(1), marker.row(0), v1);
		Core.subtract(marker.row(2), marker.row(0), v2);
		double clockwise = (v1.get(0, 0)[0] * v2.get(0, 0)[1]) - (v1.get(0, 0)[1] * v2.get(0, 0)[0]);
		if (clockwise < 0) {
			Mat tmp = new Mat();
			marker.row(1).copyTo(tmp);
			marker.row(3).copyTo(marker.row(1));
			tmp.copyTo(marker.row(3));
		}
	}

	private double getArea(Mat curve) {
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

	private double helen(double a, double b, double c) {
		double p = (a + b + c) / 2.0;
		return Math.sqrt(p * (p - a) * (p - b) * (p - c));
	}

}

package pers.season.vml.calib;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.AKAZE;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.DescriptorMatcher;
import pers.season.vml.util.ImUtils;

public class AKazeMatch {
	protected Mat srcDes;
	protected MatOfKeyPoint srcKp;
	protected Mat srcPic;

	public void setTemplate(Mat template) {
		srcDes = new Mat();
		srcKp = new MatOfKeyPoint();
		this.srcPic = template;
		AKAZE ak = AKAZE.create();
		// ak.setThreshold(threshold);
		ak.detectAndCompute(template, new Mat(), srcKp, srcDes);
	}

	public Mat getMatchingPoint(Mat pic) {
		Mat dstDes = new Mat();
		MatOfKeyPoint dstKp = new MatOfKeyPoint();
		AKAZE ak = AKAZE.create();
		ak.detectAndCompute(pic, new Mat(), dstKp, dstDes);

		DescriptorMatcher bfm = BFMatcher.create(Core.NORM_HAMMING);
		List<MatOfDMatch> matchesLst = new ArrayList<MatOfDMatch>();
		bfm.knnMatch(srcDes, dstDes, matchesLst, 2);

		Collections.sort(matchesLst, new Comparator<MatOfDMatch>() {

			@Override
			public int compare(MatOfDMatch o1, MatOfDMatch o2) {
				double k1 = o1.get(0, 0)[3] / o1.get(1, 0)[3];
				double k2 = o2.get(0, 0)[3] / o2.get(1, 0)[3];
				return Double.compare(k1, k2);
			}
		});

		MatOfPoint2f homoSrcPoints = new MatOfPoint2f();
		MatOfPoint2f homoDstPoints = new MatOfPoint2f();

		for (int i = 0; i < Math.min(matchesLst.size(), 15); i++) {
			DMatch[] dm = matchesLst.get(i).toArray();

			Point p1 = srcKp.toArray()[dm[0].queryIdx].pt;
			Point p2 = dstKp.toArray()[dm[0].trainIdx].pt;

			MatOfPoint2f t1 = new MatOfPoint2f(p1);
			MatOfPoint2f t2 = new MatOfPoint2f(p2);

			homoSrcPoints.push_back(t1);
			homoDstPoints.push_back(t2);

		}
		ImUtils.printTiming();

		Mat homo = Calib3d.findHomography(homoSrcPoints, homoDstPoints);

		Mat srcRect = Mat.ones(3, 4, CvType.CV_64F);
		srcRect.put(0, 0, 0, 0, srcPic.cols(), srcPic.cols());
		srcRect.put(1, 0, 0, srcPic.rows(), 0, srcPic.rows());

		Mat dstRect = new Mat();
		Core.gemm(homo, srcRect, 1, new Mat(), 0, dstRect);
		Core.divide(dstRect.row(0), dstRect.row(2), dstRect.row(0));
		Core.divide(dstRect.row(1), dstRect.row(2), dstRect.row(1));
		return dstRect;
	}

}

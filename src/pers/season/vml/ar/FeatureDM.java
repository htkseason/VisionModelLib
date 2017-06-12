package pers.season.vml.ar;

import java.util.ArrayList;
import java.util.List;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.features2d.AKAZE;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Feature2D;
import org.opencv.features2d.KAZE;
import org.opencv.imgproc.Imgproc;

public class FeatureDM {
	protected Feature2D f2d = AKAZE.create(AKAZE.DESCRIPTOR_KAZE, 0, 3, 0.001f, 4, 4, KAZE.DIFF_PM_G2);
	protected DescriptorMatcher bfm = BFMatcher.create(Core.NORM_HAMMING);

	protected Mat template;
	protected MatOfKeyPoint srcKp;
	protected Mat srcDes;

	protected final static int MIN_QUERY_KPS = 8;

	public void setTemplate(Mat template) {
		srcDes = new Mat();
		srcKp = new MatOfKeyPoint();
		this.template = template;
		f2d.detectAndCompute(template, new Mat(), srcKp, srcDes);
		System.out.println(srcKp.total() + " features learned.");
	}

	public Mat findHomo(Mat pic, boolean refine) {

		Mat dstDes = new Mat();
		MatOfKeyPoint dstKp = new MatOfKeyPoint();
		// sometimes UnknownException...
		f2d.detectAndCompute(pic, new Mat(), dstKp, dstDes);

		if (dstKp.total() < MIN_QUERY_KPS)
			return null;

		List<MatOfDMatch> matchesLst = new ArrayList<MatOfDMatch>();
		bfm.knnMatch(dstDes, srcDes, matchesLst, 2);

		MatOfPoint2f goodSrcPoints = new MatOfPoint2f();
		MatOfPoint2f goodDstPoints = new MatOfPoint2f();
		KeyPoint[] srcKpArr = srcKp.toArray();
		KeyPoint[] dstKpArr = dstKp.toArray();

		for (int i = 0; i < matchesLst.size(); i++) {
			DMatch[] dm = matchesLst.get(i).toArray();
			if (dm[0].distance < 0.75 * dm[1].distance) {
				Point p1 = dstKpArr[dm[0].queryIdx].pt;
				Point p2 = srcKpArr[dm[0].trainIdx].pt;

				goodSrcPoints.push_back(new MatOfPoint2f(p2));
				goodDstPoints.push_back(new MatOfPoint2f(p1));
			}

		}

		if (goodSrcPoints.total() < MIN_QUERY_KPS || goodDstPoints.total() < MIN_QUERY_KPS)
			return null;

		Mat mask = new Mat();
		Mat homo = Calib3d.findHomography(goodSrcPoints, goodDstPoints, Calib3d.RANSAC, 3, mask, 2000, 0.995);
		homo.convertTo(homo, CvType.CV_32F);
		if (homo.empty())
			return null;

		for (int i = 0; i < mask.total(); i++) {
			if (mask.get(i, 0)[0] != 0)
				Imgproc.circle(pic, goodDstPoints.toArray()[i], 2, new Scalar(0, 255, 0), 2);
		}
		int inliersCount = (int) Core.sumElems(mask).val[0];

		if (inliersCount < MIN_QUERY_KPS)
			return null;

		if (refine) {
			Mat warpedPic = new Mat();
			Imgproc.warpPerspective(pic, warpedPic, homo, template.size(),
					Imgproc.WARP_INVERSE_MAP | Imgproc.INTER_CUBIC);
			Mat refinedHomo = findHomo(warpedPic, false);
			if (refinedHomo == null)
				return homo;
			Mat result = new Mat();
			Core.gemm(homo, refinedHomo, 1.0, new Mat(), 0, result);
			return result;
		}

		return homo;
	}

	public void solvePnp(Mat homo, Mat camMat, Mat rvec, Mat tvec) {
		MatOfPoint3f srcPts = new MatOfPoint3f(new Point3(0, 0, 0), new Point3(template.width(), 0, 0),
				new Point3(template.width(), template.height(), 0), new Point3(0, template.height(), 0));
		MatOfPoint2f dstPts = new MatOfPoint2f(getQuadFromHomo(homo));
		Calib3d.solvePnP(srcPts, dstPts, camMat, new MatOfDouble(), rvec, tvec);

		rvec.convertTo(rvec, CvType.CV_32F);
		tvec.convertTo(tvec, CvType.CV_32F);

	}

	public Mat getQuadFromHomo(Mat homo) {
		Mat srcRect = Mat.ones(4, 1, CvType.CV_32FC2);
		srcRect.put(0, 0, 0, 0, template.width(), 0, template.width(), template.height(), 0, template.height());
		
		Mat dstQuad = new Mat();
		Core.perspectiveTransform(srcRect, dstQuad, homo);

		return dstQuad;
	}

}

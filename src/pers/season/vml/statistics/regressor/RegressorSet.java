package pers.season.vml.statistics.regressor;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.imgproc.Imgproc;

public class RegressorSet {
	protected static final int CORE_COUNTS = Runtime.getRuntime().availableProcessors();
	protected static ExecutorService threadPool = Executors.newCachedThreadPool();

	public static Mat track(Mat patches, Mat pic, Mat srcPts, Mat refShape, Size patchSize, Size searchSize) {

		Mat R = getPtsAffineTrans(srcPts, refShape, pic.width() / 2, pic.height() / 2);
		Mat dstPts = warpPtsAffine(srcPts, R);
		Mat affPic = pic.clone();
		Core.add(affPic, new Scalar(1), affPic);
		Core.log(affPic, affPic);
		Imgproc.warpAffine(affPic, affPic, R, pic.size());

		Semaphore sema = new Semaphore(0);
		for (int threadIndex = 0; threadIndex < CORE_COUNTS; threadIndex++) {
			final int curThreadIndex = threadIndex;
			final Mat dstPtsFinal = dstPts;
			threadPool.execute(new Runnable() {
				@Override
				public void run() {
					for (int i = 0; i < dstPtsFinal.rows() / 2; i++) {
						if (i % CORE_COUNTS != curThreadIndex)
							continue;
						double px = dstPtsFinal.get(i * 2, 0)[0];
						double py = dstPtsFinal.get(i * 2 + 1, 0)[0];
						Mat response = predictArea(affPic, patches.col(i), new Point(px, py), patchSize, searchSize);

						MinMaxLocResult mmr = Core.minMaxLoc(response);

						dstPtsFinal.put(i * 2, 0, mmr.maxLoc.x - (int) searchSize.width / 2 - 1 + px);
						dstPtsFinal.put(i * 2 + 1, 0, mmr.maxLoc.y - (int) searchSize.height / 2 - 1 + py);
					}
					sema.release();
				}
			});
		}
		try {
			sema.acquire(CORE_COUNTS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		for (int i = 0; i < dstPts.rows() / 2; i++) {
			Imgproc.circle(affPic, new Point(dstPts.get(2 * i, 0)[0], dstPts.get(2 * i + 1, 0)[0]), 2, new Scalar(255));
		}

		dstPts = reversePtsAffine(dstPts, R);
		return dstPts;
	}

	public static Mat getPtsAffineTrans(Mat srcPts, Mat dstPts, double xOffset, double yOffset) {
		Mat R = calcSimi(srcPts, dstPts);
		R.put(0, 2, R.get(0, 2)[0] + xOffset);
		R.put(1, 2, R.get(1, 2)[0] + yOffset);
		return R;
	}

	public static Mat reversePtsAffine(Mat affPts, Mat R) {
		Mat reverseR = new Mat();
		Imgproc.invertAffineTransform(R, reverseR);
		return warpPtsAffine(affPts, reverseR);
	}

	public static Mat warpPtsAffine(Mat srcPts, Mat R) {
		Mat result = srcPts.clone();
		for (int i = 0; i < srcPts.rows() / 2; i++) {
			double tpx = srcPts.get(i * 2, 0)[0];
			double tpy = srcPts.get(i * 2 + 1, 0)[0];
			double px = tpx * R.get(0, 0)[0] + tpy * R.get(0, 1)[0] + R.get(0, 2)[0];
			double py = tpx * R.get(1, 0)[0] + tpy * R.get(1, 1)[0] + R.get(1, 2)[0];
			result.put(i * 2, 0, px);
			result.put(i * 2 + 1, 0, py);
		}
		return result;
	}

	public static Mat predictArea(Mat pic, Mat theta, Point center, Size patchSize, Size searchSize) {
		Mat response = new Mat(searchSize, CvType.CV_32F);
		// 21/20-->10
		int searchHeightHalf = (int) searchSize.height / 2;
		int searchWidthHalf = (int) searchSize.width / 2;
		int patchHeightHalf = (int) patchSize.height / 2;
		int patchWidthHalf = (int) patchSize.width / 2;
		double bias = theta.get(0, 0)[0];
		theta = theta.rowRange(1, theta.rows()).clone().reshape(1, patchHeightHalf * 2 + 1);
		for (int y = -searchHeightHalf; y <= searchHeightHalf; y++) {
			for (int x = -searchWidthHalf; x <= searchWidthHalf; x++) {
				int rowStart = (int) center.y + y - patchHeightHalf;
				int rowEnd = (int) center.y + y + patchHeightHalf + 1;
				int colStart = (int) center.x + x - patchWidthHalf;
				int colEnd = (int) center.x + x + patchWidthHalf + 1;
				if (rowStart < 0 || colStart < 0 || rowEnd >= pic.height() || colEnd >= pic.width()) {
					response.put(y + searchHeightHalf, x + searchWidthHalf, Float.NaN);
					continue;
				}
				Mat subpic = pic.submat(rowStart, rowEnd, colStart, colEnd);
				Mat tempMat = new Mat();
				Core.multiply(subpic, theta, tempMat);
				Scalar rVal = Core.sumElems(tempMat);
				double r = rVal.val[0] + bias;
				response.put(y + searchHeightHalf, x + searchWidthHalf, r);
			}
		}
		return response;
	}

	protected static Mat calcSimi(Mat pts, Mat ref) {
		// compute translation
		double mx = 0, my = 0;
		double refmx = 0, refmy = 0;
		for (int i = 0; i < pts.rows() / 2; i++) {
			mx += pts.get(i * 2, 0)[0];
			my += pts.get(i * 2 + 1, 0)[0];
			refmx += ref.get(i * 2, 0)[0];
			refmy += ref.get(i * 2 + 1, 0)[0];
		}
		mx /= pts.rows() / 2;
		my /= pts.rows() / 2;
		refmx /= ref.rows() / 2;
		refmy /= ref.rows() / 2;

		double a = 0, b = 0, c = 0;
		for (int i = 0; i < pts.rows() / 2; i++) {
			double refx = ref.get(i * 2, 0)[0] - refmx, refy = ref.get(i * 2 + 1, 0)[0] - refmy;
			double ptsx = pts.get(i * 2, 0)[0] - mx, ptsy = pts.get(i * 2 + 1, 0)[0] - my;
			a += refx * refx + refy * refy;
			b += refx * ptsx + refy * ptsy;
			c += refx * ptsy - refy * ptsx;
		}
		b /= a;
		c /= a;
		double scale = Math.sqrt(b * b + c * c);
		double theta = Math.atan2(c, b);
		double sc = scale * Math.cos(theta);
		double ss = scale * Math.sin(theta);
		Mat R = new Mat(2, 3, CvType.CV_32F);
		R.put(0, 0, new double[] { sc, -ss, mx - refmx, ss, sc, my - refmy });
		Imgproc.invertAffineTransform(R, R);
		return R;
	}

}

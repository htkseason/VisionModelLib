package pers.season.vml.statistics.patch;

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

import pers.season.vml.util.ImUtils;

public class PatchSet {

	protected static ExecutorService threadPool = Executors
			.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
	public Mat[] patches;
	public Size patchSize;
	public Mat refShape;
	public int PTS_COUNT;

	public static PatchSet load(String path, String patches_name, String refShape_name, Size patchSize) {
		return load(ImUtils.loadMat(path + patches_name), ImUtils.loadMat(path + refShape_name), patchSize);
	}

	public static PatchSet load(Mat patches, Mat refShape, Size patchSize) {
		PatchSet ps = new PatchSet();

		ps.patches = new Mat[patches.cols()];
		for (int i = 0; i < patches.cols(); i++) {
			ps.patches[i] = patches.col(i).rowRange(1, patches.rows()).clone().reshape(1, (int) patchSize.height);
		}
		ps.patchSize = patchSize.clone();
		ps.refShape = refShape.clone();
		ps.PTS_COUNT = patches.cols();

		System.out.println("PatchSet inited. " + ps.PTS_COUNT + " points");
		return ps;
	}

	public Mat track(Mat pic, Mat srcPts, Size searchSize) {

		Mat R = getPtsAffineTrans(srcPts, refShape, pic.width() / 2, pic.height() / 2);
		Mat dstPts = warpPtsAffine(srcPts, R);
		Mat affPic = pic.clone();
		Core.add(affPic, new Scalar(1), affPic);
		Core.log(affPic, affPic);
		Imgproc.warpAffine(affPic, affPic, R, pic.size());

		Semaphore sema = new Semaphore(0);

		final Mat finalDstPts = dstPts;

		for (int i = 0; i < finalDstPts.rows() / 2; i++) {
			final int index = i;
			threadPool.execute(new Runnable() {
				@Override
				public void run() {
					double px = finalDstPts.get(index * 2, 0)[0];
					double py = finalDstPts.get(index * 2 + 1, 0)[0];
					Mat response = predictArea(affPic, patches[index], new Point(px, py), patchSize, searchSize);

					MinMaxLocResult mmr = Core.minMaxLoc(response);

					finalDstPts.put(index * 2, 0, mmr.maxLoc.x - response.width() / 2 + px);
					finalDstPts.put(index * 2 + 1, 0, mmr.maxLoc.y - response.height() / 2 + py);
					sema.release();
				}

			});
		}
		try {
			sema.acquire(finalDstPts.rows() / 2);
		} catch (InterruptedException e) {
			e.printStackTrace();
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
		int searchHeightHalf = (int) searchSize.height / 2;
		int searchWidthHalf = (int) searchSize.width / 2;
		int patchHeightHalf = (int) patchSize.height / 2;
		int patchWidthHalf = (int) patchSize.width / 2;

		int rowStart = (int) center.y - patchHeightHalf - searchHeightHalf;
		int rowEnd = (int) center.y + patchHeightHalf + 1 + searchHeightHalf;
		int colStart = (int) center.x - patchWidthHalf - searchWidthHalf;
		int colEnd = (int) center.x + patchWidthHalf + 1 + searchWidthHalf;
		if (rowStart < 0) {
			rowStart = 0;
		}
		if (colStart < 0) {
			colStart = 0;
		}
		if (rowEnd > pic.rows()) {
			rowEnd = pic.rows();
		}
		if (colEnd > pic.cols()) {
			colEnd = pic.cols();
		}
		if (rowStart >= rowEnd || colStart >= colEnd) {
			return new Mat();
		}

		Mat subpic = pic.submat(rowStart, rowEnd, colStart, colEnd);
		Mat response = new Mat();
		if (theta.rows() <= subpic.rows() && theta.cols() <= subpic.cols()) {
			Imgproc.matchTemplate(subpic, theta, response, Imgproc.TM_CCOEFF_NORMED);
		}

		return response;
	}

	public static Mat calcSimi(Mat pts, Mat ref) {
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

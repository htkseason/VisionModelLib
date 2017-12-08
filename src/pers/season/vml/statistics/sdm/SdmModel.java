package pers.season.vml.statistics.sdm;

import java.io.File;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.ml.LinearRegression;
import pers.season.vml.statistics.patch.PatchSet;
import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public class SdmModel {

	public Mat theta;

	protected SdmModel(Mat theta) {
		this.theta = theta;
	}

	public static SdmModel load(String dataPath) {
		Mat theta = new Mat();
		for (int i = 0; i < MuctData.getPtsCounts() * 2; i++) {
			theta.push_back(ImUtils.loadMat(dataPath+"theta_" + i).t());
		}
		return new SdmModel(theta.t().clone());
	}

	public static Mat computeFeature(Mat pic, Mat pts, Mat refShape, Size blockSize) {
		Mat R = PatchSet.getPtsAffineTrans(pts, refShape, pic.width() / 2, pic.height() / 2);
		Mat affPts = PatchSet.warpPtsAffine(pts, R);
		Mat affPic = new Mat();
		Imgproc.warpAffine(pic, affPic, R, pic.size());

		Mat result = new Mat(SdmHogDescriptor.DESCRIPTION_SIZE * affPts.rows() / 2, 1, CvType.CV_32F);
		MatOfFloat mof = new MatOfFloat();
		for (int i = 0; i < affPts.rows() / 2; i++) {
			SdmHogDescriptor.compute(affPic, new Point(affPts.get(i * 2, 0)[0], affPts.get(i * 2 + 1, 0)[0]), blockSize,
					mof);

			mof.copyTo(result.rowRange(i * SdmHogDescriptor.DESCRIPTION_SIZE,
					(i + 1) * SdmHogDescriptor.DESCRIPTION_SIZE));
		}
		// ImUtils.imshow(affPic);

		return result;
	}

	public static Mat calcResidual(Mat feature, Mat theta) {
		Mat result = new Mat();
		Core.gemm(theta.rowRange(1, theta.rows()).t(), feature, 1, new Mat(), 0, result);
		Core.add(result, theta.row(0).t(), result);
		return result;
	}
}

package pers.season.vml.statistics.sdm;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.statistics.regressor.RegressorSet;
import pers.season.vml.util.ImUtils;

public class SdmModel {

	public static Mat computeFeature(Mat pic, Mat pts, Mat refShape, Size blockSize) {

		Mat R = RegressorSet.getPtsAffineTrans(pts, refShape, pic.width() / 2, pic.height() / 2);
		Mat affPts = RegressorSet.warpPtsAffine(pts, R);
		Mat affPic = new Mat();
		Imgproc.warpAffine(pic, affPic, R, pic.size());

		Mat result = new Mat(SdmHogDescriptor.DESCRIPTION_SIZE*affPts.rows()/2,1,CvType.CV_32F);
		MatOfFloat mof = new MatOfFloat();
		for (int i = 0; i < affPts.rows() / 2; i++) {
			SdmHogDescriptor.compute(affPic, new Point(affPts.get(i * 2, 0)[0], affPts.get(i * 2 + 1, 0)[0]), blockSize,
					mof);
			
			mof.copyTo(result.rowRange(i*SdmHogDescriptor.DESCRIPTION_SIZE, (i+1)*SdmHogDescriptor.DESCRIPTION_SIZE));

		}
		//ImUtils.imshow(affPic);
		
		
		return result;
	}
}

package pers.season.vml.statistics.regressor;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

public class Regressor {

	protected Mat theta;

	public Regressor() {
		
	}
	public Regressor(Mat theta) {
		load(theta);
	}
	
	public void load(Mat theta) {
		this.theta = theta;
	}

	public Mat predictArea(Mat pic, Point center, Size patchSize, Size searchSize) {
		Mat result = new Mat(searchSize, CvType.CV_32F);
		// 21/20-->10
		int searchHeightHalf = (int) searchSize.height / 2;
		int searchWidthHalf = (int) searchSize.width / 2;
		int patchHeightHalf = (int) patchSize.height / 2;
		int patchWidthHalf = (int) patchSize.width / 2;
		for (int y = -searchHeightHalf; y <= searchHeightHalf; y++) {
			for (int x = -searchWidthHalf; x <= searchWidthHalf; x++) {
				int rowStart = (int) center.y + y - patchHeightHalf;
				int rowEnd = (int) center.y + y + patchHeightHalf + 1;
				int colStart = (int) center.x + x - patchWidthHalf;
				int colEnd = (int) center.x + x + patchWidthHalf + 1;
				if (rowStart < 0 || colStart < 0 || rowEnd >= pic.height() || colEnd >= pic.width()) {
					result.put(y + searchHeightHalf, x + searchWidthHalf, Float.NaN);
					continue;
				}
				Mat subpic = pic.submat(rowStart, rowEnd, colStart, colEnd);
				float r = predict(subpic);
				result.put(y + searchHeightHalf, x + searchWidthHalf, r);
			}
		}
		return result;
	}

	public float predict(Mat pic) {
		Mat result = new Mat();
		pic = pic.clone().reshape(1, 1);
		Core.gemm(pic, theta.rowRange(1, theta.rows()), 1, new Mat(), 0, result);
		Core.add(result, new Scalar(theta.get(0, 0)[0]), result);
		return (float) result.get(0, 0)[0];

	}
}

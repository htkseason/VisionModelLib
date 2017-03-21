package priv.season.aam.statistics.regressor;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.LogisticRegression;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;

import priv.season.aam.util.ImUtils;
import priv.season.aam.util.MuctData;

public class RegressorTrain {
	static int point = 54;

	public static void trainLR() {

		Mat sample = new Mat();
		Mat response = new Mat();
		for (int i = 0; i < 1000; i++) {
			sample.push_back(getSample(i, point, 20, 15.0, 4.0, 8));
			response.push_back(getResponse(15.0, 8));
		}
		System.out.println(sample.size());
		System.out.println(response.size());

		ImUtils.startTiming();
		LogisticRegression lr = LogisticRegression.create();
		lr.setLearningRate(0.01);
		lr.setRegularization(LogisticRegression.REG_L2);
		
		lr.setTrainMethod(LogisticRegression.MINI_BATCH);
		lr.setMiniBatchSize(10);
		lr.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 50000, 0));

		lr.train(sample, Ml.ROW_SAMPLE, response);
		System.out.println(ImUtils.getTiming());
		Mat theta = lr.get_learnt_thetas().t();

		Mat res = new Mat(getSample(0, point, 20, 15.0, 4.0, 4).rows(), 100, CvType.CV_32F);
		for (int ii = 0; ii < res.cols(); ii++) {
			Mat testSample = getSample(ii + 1000, point, 20, 15.0, 4.0, 4);
			for (int i = 0; i < res.rows(); i++) {
				double r = predict(testSample.row(i), theta);

				res.put(i, ii, r);
			}
			System.gc();
		}
		Core.normalize(res, res, 0, 255, Core.NORM_MINMAX);
		ImUtils.imshow(new JFrame(), res, 7);

		showDis(theta, 1000);
		ImUtils.imshow(MuctData.getGrayJpg(1000));
		showDis(theta, 1100);
		ImUtils.imshow(MuctData.getGrayJpg(1100));
		showDis(theta, 1200);
		ImUtils.imshow(MuctData.getGrayJpg(1200));
		showDis(theta, 1300);
		ImUtils.imshow(MuctData.getGrayJpg(1300));
		showDis(theta, 1400);
		ImUtils.imshow(MuctData.getGrayJpg(1400));
		showDis(theta, 1500);
		ImUtils.imshow(MuctData.getGrayJpg(1500));

	}

	private static void showDis(Mat theta, int number) {
		Mat pic = MuctData.getGrayJpg(number);
		pic.convertTo(pic, CvType.CV_32F);
		Core.add(pic, new Scalar(1), pic);
		Core.log(pic, pic);
		int px = (int) MuctData.getPts(number)[point * 2];
		int py = (int) MuctData.getPts(number)[point * 2 + 1];
		Mat res = new Mat(61, 61, CvType.CV_32F);
		for (int x = -30; x <= 30; x++)
			for (int y = -30; y <= 30; y++) {

				Mat test = pic.submat(py + y - 10, py + y + 10, px + x - 10, px + x + 10).clone().reshape(1, 1);

				double r = predict(test, theta);

				res.put(y + 30, x + 30, r);
			}

		Core.normalize(res, res, 0, 255, Core.NORM_MINMAX);
		ImUtils.imshow(new JFrame(), res, 5);
	}

	private static double predict(Mat sample, Mat theta) {
		Mat result = new Mat();
		Core.gemm(sample, theta.rowRange(1, theta.rows()), 1, new Mat(), 0, result);
		Core.add(result, new Scalar(theta.get(0, 0)[0]), result);
		return result.get(0, 0)[0];
	}

	private static Mat getResponse(double degreeGap, int layerCount) {
		Mat result = Mat.zeros((int) (360 / degreeGap * (layerCount - 1) + 9), 1, CvType.CV_32F);
		result.put(0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1);
		return result;
	}

	private static Mat getSample(int jpgIndex, int ptsIndex, int patchWidth, double degreeGap, double radiusGap,
			int layerCount) {
		Mat pic = MuctData.getGrayJpg(jpgIndex);
		pic.convertTo(pic, CvType.CV_32F);
		Core.add(pic, new Scalar(1), pic);
		Core.log(pic, pic);
		int px = (int) MuctData.getPts(jpgIndex)[ptsIndex * 2];
		int py = (int) MuctData.getPts(jpgIndex)[ptsIndex * 2 + 1];
		Mat sample = new Mat();
		for (int x = -1; x <= 1; x++)
			for (int y = -1; y <= 1; y++)
				sample.push_back(pic.submat(py + y - patchWidth / 2, py + y + patchWidth / 2, px + x - patchWidth / 2,
						px + x + patchWidth / 2).clone().reshape(1, 1));

		for (int layer = 1; layer < layerCount; layer++) {
			for (int degree = 0; degree < 360; degree += degreeGap) {
				int x = (int) (px + layer * radiusGap * Math.cos(degree / 180.0 * Math.PI));
				int y = (int) (py + layer * radiusGap * Math.sin(degree / 180.0 * Math.PI));
				sample.push_back(
						pic.submat(y - patchWidth / 2, y + patchWidth / 2, x - patchWidth / 2, x + patchWidth / 2)
								.clone().reshape(1, 1));
			}
		}

		return sample;
	}
}

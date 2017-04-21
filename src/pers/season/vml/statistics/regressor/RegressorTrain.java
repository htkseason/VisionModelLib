package pers.season.vml.statistics.regressor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.LogisticRegression;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;

import pers.season.vml.statistics.shape.ShapeInstance;
import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public class RegressorTrain {

	public static Mat refShape;

	public static Mat trainLinearModel(int point, Size patchSize, Size searchSize, int samplingGap,
			double trainSampleProportion, LearningParams lp) {
		Mat sample = new Mat();
		Mat response = new Mat();

		for (int i = 0; i < MuctData.getSize(); i++) {
			if (Math.random() > trainSampleProportion)
				continue;

			Mat s = new Mat();
			Mat r = new Mat();
			getSample(s, r, i, point, patchSize, searchSize, samplingGap);
			sample.push_back(s);
			response.push_back(r);

			System.gc();
		}
		System.out.println(sample.rows() + " samples loaded");
		
		LinearRegression lr = new LinearRegression();
		lr.setData(sample, response, 1);

		double preCost = Double.MAX_VALUE;
		double delta = lp.initLearningRate;
		for (int i = 0; i < lp.iteration; i++) {
			Mat g_ori = lr.getGradient(lp.batchSize);
			Mat g = new Mat();
			Core.multiply(g_ori, new Scalar(delta), g);
			Core.subtract(lr.theta, g, lr.theta);
			System.gc();
			if (i % lp.learningRateCheckStep == 0) {
				double cost = lr.getCost();
				if (cost > preCost) {
					delta /= lp.learningRateDescentRatio;
					System.out.println("learning rate  " + delta * lp.learningRateDescentRatio + " --> " + delta);
				}
				preCost = cost;
				System.out.println("iter = " + i + "/" + lp.iteration + "\t\tcost = " + preCost);
			}

		}
		
		return lr.theta;

	}

	public static void debug_measure(int point, Mat theta) {
		JFrame win = new JFrame();
		// Mat resResult = new Mat();
		Regressor reg = new Regressor(theta);
		
		Mat maxloc = new Mat();

		for (int i = 3500; i < 4000; i++) {

			Mat pic = MuctData.getGrayJpg(i);
			pic.convertTo(pic, CvType.CV_32F);
			Core.add(pic, new Scalar(1), pic);
			Core.log(pic, pic);
			// Core.normalize(pic, pic,-1,1,Core.NORM_MINMAX);
			int tpx = (int) MuctData.getPts(i)[point * 2];
			int tpy = (int) MuctData.getPts(i)[point * 2 + 1];
			Mat R = calcSimi(MuctData.getPtsMat(i), refShape);
			R.put(0, 2, R.get(0, 2)[0] + pic.width() / 2);
			R.put(1, 2, R.get(1, 2)[0] + pic.height() / 2);
			int px = (int) Math.round((tpx * R.get(0, 0)[0] + tpy * R.get(0, 1)[0] + R.get(0, 2)[0]));
			int py = (int) Math.round((tpx * R.get(1, 0)[0] + tpy * R.get(1, 1)[0] + R.get(1, 2)[0]));
			Imgproc.warpAffine(pic, pic, R, pic.size());
			
			Mat r = reg.predictArea(pic, new Point(px, py), new Size(41, 41), new Size(61, 61));
			
			
			Imgproc.blur(r, r, new Size(5, 5), new Point(-1, -1), Core.BORDER_CONSTANT);
			// Imgproc.GaussianBlur(r, r, new Size(7,7),1, 1,
			// Core.BORDER_CONSTANT);

			
			// ImUtils.imshow(win, r, 5);
			MinMaxLocResult mmr = Core.minMaxLoc(r);
			Mat rmat = new Mat(1, 2, CvType.CV_32F);

			rmat.put(0, 0, mmr.maxLoc.x, mmr.maxLoc.y);
			maxloc.push_back(rmat);
			System.gc();
			// resResult.push_back(r);

		}
		ImUtils.saveMat(maxloc, "e:/maxloc");

		ImUtils.saveMat(theta, "e:/theta");
	}

	public static void getSample(Mat sample, Mat response, int sampleIndex, int ptsIndex, Size patchSize,
			Size searchSize, int samplingGap) {

		Mat pic = MuctData.getGrayJpg(sampleIndex);
		pic.convertTo(pic, CvType.CV_32F);
		Core.add(pic, new Scalar(1), pic);
		Core.log(pic, pic);
		int patchWidthHalf = (int) patchSize.width / 2;
		int patchHeightHalf = (int) patchSize.height / 2;
		int searchWidthHalf = (int) searchSize.width / 2;
		int searchHeightHalf = (int) searchSize.height / 2;

		int tpx = (int) MuctData.getPts(sampleIndex)[ptsIndex * 2];
		int tpy = (int) MuctData.getPts(sampleIndex)[ptsIndex * 2 + 1];
		if (tpx - searchWidthHalf - patchWidthHalf < 0 || tpx + searchWidthHalf + patchWidthHalf >= pic.width()
				|| tpy - searchHeightHalf - patchHeightHalf < 0
				|| tpy + searchHeightHalf + patchHeightHalf >= pic.height())
			return;

		Mat R = calcSimi(MuctData.getPtsMat(sampleIndex), refShape);
		R.put(0, 2, R.get(0, 2)[0] + pic.width() / 2);
		R.put(1, 2, R.get(1, 2)[0] + pic.height() / 2);
		int px = (int) Math.round((tpx * R.get(0, 0)[0] + tpy * R.get(0, 1)[0] + R.get(0, 2)[0]));
		int py = (int) Math.round((tpx * R.get(1, 0)[0] + tpy * R.get(1, 1)[0] + R.get(1, 2)[0]));
		Imgproc.warpAffine(pic, pic, R, pic.size());

		int posSize = 2;
		for (int y = -posSize; y <= posSize; y++) {
			for (int x = -posSize; x <= posSize; x++) {
				Mat s = pic.submat(py + y - patchHeightHalf, py + y + patchHeightHalf + 1, px + x - patchWidthHalf,
						px + x + patchWidthHalf + 1).clone().reshape(1, 1);

				sample.push_back(s);
				Mat r = new Mat(1, 1, CvType.CV_32F);
				double dev = 1;
				r.put(0, 0, Math.exp(-(x * x + y * y) / (dev * dev * 2)));

				// System.out.println(r.get(0, 0)[0]);

				response.push_back(r);
			}
		}
		for (int y = -searchHeightHalf; y <= searchHeightHalf; y += samplingGap) {
			for (int x = -searchWidthHalf; x <= searchWidthHalf; x += samplingGap) {
				if (Math.abs(x) <= posSize && Math.abs(y) <= posSize) {
					continue;
				}
				Mat s = pic.submat(py + y - patchHeightHalf, py + y + patchHeightHalf + 1, px + x - patchWidthHalf,
						px + x + patchWidthHalf + 1).clone().reshape(1, 1);

				sample.push_back(s);
				Mat r = new Mat(1, 1, CvType.CV_32F);
				r.put(0, 0, 0);
				response.push_back(r);

			}
		}

	}

	public static void init(int width, int height) {
		refShape = new Mat();
		// calculate mean shape
		Mat shapes = new Mat();
		for (int i = 0; i < MuctData.getSize(); i++)
			shapes.push_back(MuctData.getPtsMat(i).t());
		shapes = shapes.t();

		Mat mx = Mat.zeros(1, shapes.cols(), CvType.CV_32F);
		Mat my = Mat.zeros(1, shapes.cols(), CvType.CV_32F);
		for (int i = 0; i < shapes.rows() / 2; i++) {
			Core.add(mx, shapes.row(i * 2), mx);
			Core.add(my, shapes.row(i * 2 + 1), my);
		}
		Core.divide(mx, new Scalar(shapes.rows() / 2), mx);
		Core.divide(my, new Scalar(shapes.rows() / 2), my);
		for (int i = 0; i < shapes.rows() / 2; i++) {
			Core.subtract(shapes.row(i * 2), mx, shapes.row(i * 2));
			Core.subtract(shapes.row(i * 2 + 1), my, shapes.row(i * 2 + 1));
		}

		Core.gemm(shapes, Mat.ones(shapes.cols(), 1, CvType.CV_32F), 1, new Mat(), 0, refShape);
		Core.divide(refShape, new Scalar(shapes.cols()), refShape);

		// resize refShape by width and height
		Mat shapeX = new Mat();
		Mat shapeY = new Mat();
		for (int i = 0; i < refShape.rows() / 2; i++) {
			shapeX.push_back(refShape.row(i * 2));
			shapeY.push_back(refShape.row(i * 2 + 1));
		}
		Core.normalize(shapeX, shapeX, -width / 2, width / 2, Core.NORM_MINMAX);
		Core.normalize(shapeY, shapeY, -height / 2, height / 2, Core.NORM_MINMAX);
		for (int i = 0; i < refShape.rows() / 2; i++) {
			shapeX.row(i).copyTo(refShape.row(i * 2));
			shapeY.row(i).copyTo(refShape.row(i * 2 + 1));
		}

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

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
import org.opencv.ml.LogisticRegression;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;

import pers.season.vml.statistics.shape.ShapeInstance;
import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public class RegressorTrain {
	static int point = 54;
	static Mat refShape = new Mat();
	static {
		// calculate mean shape
		Mat shapes = new Mat();
		for (int i = 0; i < MuctData.getSize(); i++)
			shapes.push_back(MuctData.getPtsMat(i).t());
		shapes = shapes.t();

		Mat mx = Mat.zeros(1, shapes.cols(), CvType.CV_64F);
		Mat my = Mat.zeros(1, shapes.cols(), CvType.CV_64F);
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

		Core.gemm(shapes, Mat.ones(shapes.cols(), 1, CvType.CV_64F), 1, new Mat(), 0, refShape);
		Core.divide(refShape, new Scalar(shapes.cols()), refShape);

		// normalize meanShape
		Core.normalize(refShape, refShape, 0, 200, Core.NORM_MINMAX);
		double mmx = 0, mmy = 0;
		for (int i = 0; i < refShape.rows() / 2; i++) {
			mmx += refShape.get(i * 2, 0)[0];
			mmy += refShape.get(i * 2 + 1, 0)[0];
		}
		mmx /= refShape.rows() / 2;
		mmy /= refShape.rows() / 2;
		for (int i = 0; i < refShape.rows() / 2; i++) {
			refShape.put(i * 2, 0, refShape.get(i * 2, 0)[0] - mmx);
			refShape.put(i * 2 + 1, 0, refShape.get(i * 2 + 1, 0)[0] - mmy);
		}

	}

	public static void trainLineR() {
		Mat theta;
		if (new File("e:/theta0").exists()) {
			theta = ImUtils.loadMat("e:/theta");
			theta.convertTo(theta, CvType.CV_32F);
		} else {
			Mat sample = new Mat();
			Mat response = new Mat();
			for (int i = 0; i < 2000; i++) {
				Mat s = new Mat();
				Mat r = new Mat();
				getSample(s, r, i, point, 20, 2, 15);
				sample.push_back(s);
				response.push_back(r);

				System.out.println(i + "/" + MuctData.getSize());
			}
			System.out.println(sample.size());
			System.out.println(response.size());
			LinearRegression lr = new LinearRegression();
			lr.setData(sample, response, 1);
			Mat costPlot = new Mat();
			double delta = 10;
			double preCost = lr.getCost();
			double blockVal = delta;
			double altProp = 1.05;
			
			for (int i = 0; i < 10000; i++) {
				ImUtils.startTiming();
				Mat g_ori = lr.getGradient(i,10000);
				Mat g = new Mat();
				Core.multiply(g_ori, new Scalar(delta), g);
				Core.subtract(lr.theta, g, lr.theta);
				double cost = lr.getCost();
				if (cost >= preCost) {
					Core.add(lr.theta, g, lr.theta);
					//blockVal = delta / altProp;
					delta = delta / 2;
					System.out.println(i + "\tblock=" + blockVal);
				} else {

					Mat ct = new Mat(1, 1, CvType.CV_64F);
					ct.put(0, 0, cost);
					costPlot.push_back(ct);

					System.out.println(
							i + "\tcost=" + cost + "\tdelta=" + delta + "\ttime=" + ImUtils.getTiming());
					preCost = cost;
					if (delta * altProp < blockVal)
						delta *= altProp;
					blockVal += delta / 500;
				}

				System.gc();
			}
			theta = lr.theta;
			ImUtils.saveMat(costPlot, "e:/costplot");
		}
		JFrame win = new JFrame();
		// Mat resResult = new Mat();
		Mat maxloc = new Mat();
		int searchSize = 30;
		for (int i = 2000; i < 3000; i++) {
			try {
				Mat r = getRes(theta, i, searchSize);

				Imgproc.blur(r, r, new Size(5, 5), new Point(-1, -1), Core.BORDER_CONSTANT);
				// Imgproc.GaussianBlur(r, r, new Size(7,7),1, 1,
				// Core.BORDER_CONSTANT);
				ImUtils.imshow(win, r, 5);
				MinMaxLocResult mmr = Core.minMaxLoc(r);
				Mat rmat = new Mat(1, 1, CvType.CV_64F);
				double dis = Math.sqrt((mmr.maxLoc.x - searchSize / 2) * (mmr.maxLoc.x - searchSize / 2)
						+ (mmr.maxLoc.y - searchSize / 2) * (mmr.maxLoc.y - searchSize / 2));
				rmat.put(0, 0, dis);
				maxloc.push_back(rmat);

				// resResult.push_back(r);
			} catch (CvException e) {
				System.out.println(i + "skipped");
				continue;
			}
		}
		ImUtils.saveMat(maxloc, "e:/maxloc");

		theta.convertTo(theta, CvType.CV_64F);
		ImUtils.saveMat(theta, "e:/theta");

	}

	public static void trainLR() {
		Mat theta;
		if (new File("e:/theta0").exists()) {
			theta = ImUtils.loadMat("e:/theta");
			theta.convertTo(theta, CvType.CV_32F);
		} else {
			Mat sample = new Mat();
			Mat response = new Mat();
			for (int i = 0; i < 1000; i++) {
				Mat s = new Mat();
				Mat r = new Mat();
				getSample(s, r, i, point, 20, 2, 15);
				sample.push_back(s);
				response.push_back(r);

				System.out.println(i + "/" + MuctData.getSize());
			}

			System.out.println(sample.size());
			System.out.println(response.size());

			ImUtils.startTiming();
			LogisticRegression lr = LogisticRegression.create();
			lr.setLearningRate(0.1);
			lr.setRegularization(LogisticRegression.REG_L2);

			lr.setTrainMethod(LogisticRegression.MINI_BATCH);
			lr.setMiniBatchSize(100);
			lr.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, (int) 1e5, -1));

			lr.train(sample, Ml.ROW_SAMPLE, response);
			System.out.println(ImUtils.getTiming() / 1000 / 60 + "min");

			theta = lr.get_learnt_thetas().t();
		}

		JFrame win = new JFrame();
		// Mat resResult = new Mat();
		Mat maxloc = new Mat();
		int searchSize = 30;
		for (int i = 1000; i < 2000; i++) {
			try {
				Mat r = getRes(theta, i, searchSize);
				Imgproc.medianBlur(r, r, 5);
				ImUtils.imshow(win, r, 5);
				MinMaxLocResult mmr = Core.minMaxLoc(r);
				Mat rmat = new Mat(1, 1, CvType.CV_64F);
				double dis = Math.sqrt((mmr.maxLoc.x - searchSize / 2) * (mmr.maxLoc.x - searchSize / 2)
						+ (mmr.maxLoc.y - searchSize / 2) * (mmr.maxLoc.y - searchSize / 2));
				rmat.put(0, 0, dis);
				maxloc.push_back(rmat);

				// resResult.push_back(r);
			} catch (CvException e) {
				System.out.println(i + "skipped");
				continue;
			}
		}
		ImUtils.saveMat(maxloc, "e:/maxloc");
		theta.convertTo(theta, CvType.CV_64F);
		ImUtils.saveMat(theta, "e:/theta");
	}

	private static Mat getRes(Mat theta, int testIndex, int searchSize) {
		Mat pic = MuctData.getGrayJpg(testIndex);
		pic.convertTo(pic, CvType.CV_32F);
		Core.add(pic, new Scalar(1), pic);
		Core.log(pic, pic);
		// Core.normalize(pic, pic,-1,1,Core.NORM_MINMAX);
		int tpx = (int) MuctData.getPts(testIndex)[point * 2];
		int tpy = (int) MuctData.getPts(testIndex)[point * 2 + 1];
		Mat R = calcSimi(MuctData.getPtsMat(testIndex), refShape);
		R.put(0, 2, R.get(0, 2)[0] + 240);
		R.put(1, 2, R.get(1, 2)[0] + 320);
		int px = (int) Math.round((tpx * R.get(0, 0)[0] + tpy * R.get(0, 1)[0] + R.get(0, 2)[0]));
		int py = (int) Math.round((tpx * R.get(1, 0)[0] + tpy * R.get(1, 1)[0] + R.get(1, 2)[0]));
		Imgproc.warpAffine(pic, pic, R, pic.size());

		int patchSize = (int) Math.round(Math.sqrt(theta.rows() - 1)) - 1;
		Mat res = new Mat(searchSize + 1, searchSize + 1, CvType.CV_32F);
		for (int x = -searchSize / 2; x <= searchSize / 2; x++)
			for (int y = -searchSize / 2; y <= searchSize / 2; y++) {

				Mat test = pic.submat(py + y - patchSize / 2, py + y + patchSize / 2 + 1, px + x - patchSize / 2,
						px + x + patchSize / 2 + 1).clone().reshape(1, 1);

				double r = predict(test, theta);

				res.put(y + searchSize / 2, x + searchSize / 2, r);
			}

		Core.normalize(res, res, 0, 255, Core.NORM_MINMAX);
		return res;
	}

	private static double predict(Mat sample, Mat theta) {
		Mat result = new Mat();
		Core.normalize(sample, sample);
		Core.gemm(sample, theta.rowRange(1, theta.rows()), 1, new Mat(), 0, result);
		Core.add(result, new Scalar(theta.get(0, 0)[0]), result);
		return result.get(0, 0)[0];
	}

	private static void getSample(Mat sample, Mat response, int sampleIndex, int ptsIndex, int patchSize, int layerGap,
			int layerCount) {
		Mat pic = MuctData.getGrayJpg(sampleIndex);
		pic.convertTo(pic, CvType.CV_32F);
		Core.add(pic, new Scalar(1), pic);
		Core.log(pic, pic);

		// Core.multiply(pic, new Scalar(20), pic);
		int tpx = (int) MuctData.getPts(sampleIndex)[ptsIndex * 2];
		int tpy = (int) MuctData.getPts(sampleIndex)[ptsIndex * 2 + 1];
		if (tpx - layerGap * layerCount - patchSize / 2 < 0
				|| tpx + layerGap * layerCount + patchSize / 2 >= pic.width()
				|| tpy - layerGap * layerCount - patchSize / 2 < 0
				|| tpy + layerGap * layerCount + patchSize / 2 >= pic.height())
			return;

		// Mat R = calcSimi(MuctData.getPtsMat(sampleIndex), refShape);
		// R.put(0, 2, R.get(0, 2)[0]+240);
		// R.put(1, 2, R.get(1, 2)[0]+320);
		// int px = (int) Math.round((tpx * R.get(0, 0)[0] + tpy * R.get(0,
		// 1)[0] + R.get(0, 2)[0]));
		// int py = (int) Math.round((tpx * R.get(1, 0)[0] + tpy * R.get(1,
		// 1)[0]+ R.get(1, 2)[0]));
		// Imgproc.warpAffine(pic, pic, R, pic.size());
		int px = tpx;
		int py = tpy;

		int posSize = 3;
		for (int y = -posSize; y <= posSize; y++) {
			for (int x = -posSize; x <= posSize; x++) {
				Mat s = pic.submat(py + y - patchSize / 2, py + y + patchSize / 2 + 1, px + x - patchSize / 2,
						px + x + patchSize / 2 + 1).clone().reshape(1, 1);

				sample.push_back(s);
				Mat r = new Mat(1, 1, CvType.CV_32F);
				double dev = 2;
				r.put(0, 0, Math.exp(-(x * x + y * y) / (dev * dev * 2)));
				//System.out.println(r.get(0, 0)[0]);

				response.push_back(r);
			}
		}
		for (int y = -layerGap * layerCount; y <= layerGap * layerCount; y += layerGap) {
			for (int x = -layerGap * layerCount; x <= layerGap * layerCount; x += layerGap) {
				if (Math.abs(x) <= posSize && Math.abs(y) <= posSize) {
					continue;
				}
				Mat s = pic.submat(py + y - patchSize / 2, py + y + patchSize / 2 + 1, px + x - patchSize / 2,
						px + x + patchSize / 2 + 1).clone().reshape(1, 1);

				sample.push_back(s);
				Mat r = new Mat(1, 1, CvType.CV_32F);
				r.put(0, 0, 0);
				response.push_back(r);

			}
		}

	}

	private static Mat calcSimi(Mat pts, Mat ref) {
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
		Mat R = new Mat(2, 3, CvType.CV_64F);
		R.put(0, 0, new double[] { sc, -ss, mx - refmx, ss, sc, my - refmy });
		Imgproc.invertAffineTransform(R, R);
		return R;
	}

}

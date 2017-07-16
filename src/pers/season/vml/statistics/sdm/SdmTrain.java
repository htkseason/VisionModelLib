package pers.season.vml.statistics.sdm;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.RTrees;

import pers.season.vml.ml.LearningParams;
import pers.season.vml.ml.LinearRegression;
import pers.season.vml.statistics.regressor.RegressorSet;
import pers.season.vml.statistics.shape.ShapeInstance;
import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public class SdmTrain {

	public static void train(Mat refShape, Size patchSize) {
		ShapeModel sm = ShapeModel.load("models/shape/", "V", "Z_e");
		ShapeInstance shape = new ShapeInstance(sm);
		JFrame win = new JFrame();

		Mat inputData = Mat.zeros(SdmHogDescriptor.DESCRIPTION_SIZE * refShape.rows() / 2, MuctData.getSize(), CvType.CV_32F);
		Mat outputData = Mat.zeros(refShape.rows(), MuctData.getSize(), CvType.CV_32F);
		MatOfFloat mof = new MatOfFloat();
		for (int i = 0; i < MuctData.getSize(); i++) {
			Mat srcPic = MuctData.getGrayJpg(i);
			Mat srcPts = MuctData.getPtsMat(i);
			Mat R = RegressorSet.getPtsAffineTrans(srcPts, refShape, srcPic.width() / 2, srcPic.height() / 2);
			Mat affPts = RegressorSet.warpPtsAffine(srcPts, R);
			Mat affPic = new Mat();
			Imgproc.warpAffine(srcPic, affPic, R, srcPic.size());

			// project non-rigid transformation
			shape.setFromPts(affPts);
			shape.Z.rowRange(4, shape.Z.rows()).setTo(new Scalar(0));
			// random orientation
			shape.setRadian((Math.random() * 2 - 1) * 30 / 180 * Math.PI);
			// random scale
			double scale = shape.getScale();
			scale = scale * (Math.random() * 0.6 + 0.7);
			shape.setScale(scale);
			// random location
			Point loc = shape.getOffset();
			double offsetAngle = Math.random() * 2 * Math.PI;
			double offsetDistance = Math.random() * 0.4 * (scale / sm.getScalePerPixel());
			shape.setOffset(new Point(loc.x + Math.cos(offsetAngle) * offsetDistance,
					loc.y + Math.sin(offsetAngle) * offsetDistance));

			Mat affPtsFalse = shape.getX();
			Core.subtract(affPts, affPtsFalse, outputData.submat(0, outputData.rows(), i, i + 1));

			for (int p = 0; p < affPtsFalse.rows() / 2; p++) {
				SdmHogDescriptor.compute(affPic,
						new Point(affPtsFalse.get(p * 2, 0)[0], affPtsFalse.get(p * 2 + 1, 0)[0]), patchSize, mof);
				mof.copyTo(inputData.submat(p * SdmHogDescriptor.DESCRIPTION_SIZE,
						(p + 1) * SdmHogDescriptor.DESCRIPTION_SIZE, i, i + 1));
				// Imgproc.circle(affPic, new Point(affPtsFalse.get(p * 2, 0)[0],
				// affPtsFalse.get(p * 2 + 1, 0)[0]), 2, new Scalar(255));
				// Imgproc.rectangle(affPic, new Point(affPtsFalse.get(p * 2, 0)[0]-12,
				// affPtsFalse.get(p * 2 + 1, 0)[0]-12),
				// new Point(affPtsFalse.get(p * 2, 0)[0]+12, affPtsFalse.get(p * 2 + 1,
				// 0)[0]+12), new Scalar(255));
			}

			// ImUtils.imshow(win, affPic, 1);

		}
		System.out.println(inputData.size());
		System.out.println(outputData.size());
		

		LinearRegression lr = new LinearRegression();
		lr.setData(inputData.t(), outputData.row(0).t(), 0, false);
		double delta = 0.001;
		for (int i = 0; i < 300000; i++) {
			Mat g_ori = lr.getGradient(100);
			Mat g = new Mat();
			Core.multiply(g_ori, new Scalar(delta), g);
			Core.subtract(lr.theta, g, lr.theta);
			System.gc();
			if (i % 1000 == 0) {
				double cost = lr.getCost();
				System.out.println("iter = " + i + "\t\tcost = " + cost);
			}
		}
		ImUtils.saveMat(lr.theta, "models/sdm/theta_0");

	}

}

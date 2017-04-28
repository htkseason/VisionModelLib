package pers.season.vml.statistics.shape;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Scalar;

import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public final class ShapeModelTrain {

	public static void visualize(ShapeModel sm) {
		JFrame win = new JFrame();
		ShapeInstance shapeModel = new ShapeInstance(sm);
		shapeModel.setFromParams(300, 0, 250, 250);
		for (int feature = 4; feature < sm.Z_SIZE; feature++) {
			win.setTitle("Feature = " + feature);
			double[] seq = new double[] { 0, 3, -3, 0 };
			for (int s = 0; s < seq.length - 1; s++) {
				for (double i = seq[s]; Math.abs(i - seq[s + 1]) > 0.001; i += 0.1 * Math.signum(seq[s + 1] - seq[s])) {
					shapeModel.Z.put(feature, 0, sm.e.get(feature, 0)[0] * i * shapeModel.getScale());
					Mat canvas = Mat.zeros(500, 500, CvType.CV_32F);
					shapeModel.printTo(canvas);
					ImUtils.imshow(win, canvas, 1);
					try {
						Thread.sleep(2);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}

		}
	}

	public static void train(String outputDir, double fractionRemain, boolean saveTransitionalData) {
		System.out.println("training shape model ...");

		// load shape
		Mat orgShapes = new Mat();
		for (int i = 0; i < MuctData.getSize(); i++) {
			orgShapes.push_back(MuctData.getPtsMat(i).t());
		}
		orgShapes = orgShapes.t();
		int sampleCounts = orgShapes.cols();
		int pointCounts = orgShapes.rows() / 2;
		System.out.println(pointCounts + " points each sample.");
		System.out.println(sampleCounts + " shape samples loaded.");
		if (saveTransitionalData)
			ImUtils.saveMat(orgShapes, outputDir + "shapes_origin");

		// duplicate a copy for training
		Mat shapes = new Mat();
		orgShapes.copyTo(shapes);

		// move to center
		Mat mx = Mat.zeros(1, sampleCounts, CvType.CV_32F);
		Mat my = Mat.zeros(1, sampleCounts, CvType.CV_32F);
		for (int i = 0; i < pointCounts; i++) {
			Core.add(mx, shapes.row(i * 2), mx);
			Core.add(my, shapes.row(i * 2 + 1), my);
		}
		Core.divide(mx, new Scalar(pointCounts), mx);
		Core.divide(my, new Scalar(pointCounts), my);
		for (int i = 0; i < pointCounts; i++) {
			Core.subtract(shapes.row(i * 2), mx, shapes.row(i * 2));
			Core.subtract(shapes.row(i * 2 + 1), my, shapes.row(i * 2 + 1));
		}

		// procrustes analysis to mean shape
		Mat preMeanShape = Mat.zeros(pointCounts * 2, 1, CvType.CV_32F);
		Mat meanShape = new Mat();
		int iter = 0;
		while (true) {
			iter++;
			Core.gemm(shapes, Mat.ones(sampleCounts, 1, CvType.CV_32F), 1, new Mat(), 0, meanShape);
			Core.divide(meanShape, new Scalar(sampleCounts), meanShape);
			Core.normalize(meanShape, meanShape);
			System.out.println("after iter = " + iter + ", procrusters cost = " + Core.norm(preMeanShape, meanShape));
			if (Core.norm(preMeanShape, meanShape) < 1e-5 || iter > 20) {
				System.out.println(
						"end with iter = " + iter + ", procrusters cost = " + Core.norm(preMeanShape, meanShape));
				break;
			}
			preMeanShape = meanShape.clone();
			for (int i = 0; i < sampleCounts; i++) {
				Mat rotateMat = getAlignRotateMat(shapes.col(i), meanShape);
				for (int j = 0; j < pointCounts; j++) {
					double x = shapes.get(2 * j, i)[0];
					double y = shapes.get(2 * j + 1, i)[0];
					shapes.put(2 * j, i, rotateMat.get(0, 0)[0] * x + rotateMat.get(0, 1)[0] * y);
					shapes.put(2 * j + 1, i, rotateMat.get(1, 0)[0] * x + rotateMat.get(1, 1)[0] * y);
				}
			}
		}

		Core.gemm(shapes, Mat.ones(sampleCounts, 1, CvType.CV_32F), 1, new Mat(), 0, meanShape);
		Core.divide(meanShape, new Scalar(sampleCounts), meanShape);
		if (saveTransitionalData) {
			ImUtils.saveMat(shapes, outputDir + "shapes_align");
			ImUtils.saveMat(meanShape, outputDir + "meanShape");
		}

		// calculate rigid matrix
		Mat rigidMat = new Mat(2 * pointCounts, 4, CvType.CV_32F);
		for (int i = 0; i < pointCounts; i++) {
			rigidMat.put(2 * i, 0, meanShape.get(2 * i, 0)[0]);
			rigidMat.put(2 * i + 1, 0, meanShape.get(2 * i + 1, 0)[0]);
			rigidMat.put(2 * i, 1, -meanShape.get(2 * i + 1, 0)[0]);
			rigidMat.put(2 * i + 1, 1, meanShape.get(2 * i, 0)[0]);
			rigidMat.put(2 * i, 2, 1.0);
			rigidMat.put(2 * i + 1, 2, 0.0);
			rigidMat.put(2 * i, 3, 0.0);
			rigidMat.put(2 * i + 1, 3, 1.0);
		}
		rigidMat = gramSchmidtOrthonormalization(rigidMat);
		if (saveTransitionalData)
			ImUtils.saveMat(rigidMat, outputDir + "rigidMat");

		// project out rigid-trans
		Mat nonRigidShape = new Mat();
		Mat shapeRigidParam = new Mat();
		Core.gemm(rigidMat.t(), shapes, 1, new Mat(), 0, shapeRigidParam);
		Mat shapeRigid = new Mat();
		Core.gemm(rigidMat, shapeRigidParam, 1, new Mat(), 0, shapeRigid);
		Core.subtract(shapes, shapeRigid, nonRigidShape);
		if (saveTransitionalData)
			ImUtils.saveMat(nonRigidShape, outputDir + "shapes_non_rigid");

		// apply svd
		System.out.println("applying svd ...");
		Mat U = new Mat();
		Mat S = new Mat();
		Mat covar = new Mat();
		Core.gemm(nonRigidShape, nonRigidShape.t(), 1, new Mat(), 0, covar);
		Core.SVDecomp(covar, S, U, new Mat());
		// last 4 vectors refer to rigid transformation, delete them manually
		U = U.colRange(0, U.cols() - 4);
		S = S.rowRange(0, S.rows() - 4);
		ImUtils.saveMat(S, outputDir + "S");
		if (saveTransitionalData) {
			ImUtils.saveMat(U, outputDir + "U_full");
		}
		System.out.println("svd completed");

		// compress data according to fraction given
		for (int i = 0; i < S.rows(); i++) {
			double frac = Core.sumElems(S.rowRange(0, i)).val[0] / Core.sumElems(S).val[0];
			if (frac >= fractionRemain) {
				U = U.colRange(0, i);
				break;
			}
		}
		System.out
				.println("compress rate : " + Core.sumElems(S.rowRange(0, U.cols())).val[0] / Core.sumElems(S).val[0]);
		System.out.println("vectors remain : " + U.cols() + "/" + S.rows());

		// put rigid-mat and svd_U together
		Mat V = Mat.zeros(pointCounts * 2, 4 + U.cols(), CvType.CV_32F);
		rigidMat.copyTo(V.colRange(0, 4));
		U.copyTo(V.colRange(4, V.cols()));
		ImUtils.saveMat(V, outputDir + "V");

		// calculate Z
		System.out.println("calculating Z and mean/stddev ...");
		Mat Z = new Mat();
		Core.gemm(V.t(), orgShapes, 1, new Mat(), 0, Z);
		if (saveTransitionalData)
			ImUtils.saveMat(Z, outputDir + "Z_unscaled");
		for (int i = 0; i < sampleCounts; i++) {
			double a = Z.get(0, i)[0];
			double b = Z.get(1, i)[0];
			double scale = Math.sqrt(a * a + b * b);
			Core.divide(Z.col(i).rowRange(4, Z.rows()), new Scalar(scale), Z.col(i).rowRange(4, Z.rows()));
		}
		if (saveTransitionalData)
			ImUtils.saveMat(Z, outputDir + "Z");

		// calculate mean/stddev of Z
		Mat z_mean = new Mat(Z.rows(), 1, CvType.CV_32F);
		Mat z_stddev = new Mat(Z.rows(), 1, CvType.CV_32F);
		for (int i = 0; i < Z.rows(); i++) {
			MatOfDouble tmean = new MatOfDouble();
			MatOfDouble tstddev = new MatOfDouble();
			Core.meanStdDev(Z.row(i), tmean, tstddev);
			if (i < 4) {
				z_mean.put(i, 0, -1);
				z_stddev.put(i, 0, -1);
			} else {
				z_mean.put(i, 0, tmean.get(0, 0)[0]);
				z_stddev.put(i, 0, tstddev.get(0, 0)[0]);
			}
		}

		ImUtils.saveMat(z_stddev, outputDir + "Z_e");
		if (saveTransitionalData)
			ImUtils.saveMat(z_mean, outputDir + "Z_mean");

		System.out.println("done!");
	}

	private static Mat gramSchmidtOrthonormalization(Mat src) {
		Mat dst = src.clone();
		for (int i = 0; i < dst.cols(); i++) {
			Mat r = dst.col(i);
			for (int j = 0; j < i; j++) {
				Mat b = dst.col(j);
				// r -= b * (b.t() * r);
				Mat temp = new Mat();
				Core.gemm(b.t(), r, 1, new Mat(), 0, temp);
				Core.gemm(b, temp, 1, new Mat(), 0, temp);
				Core.subtract(r, temp, r);
			}
			Core.normalize(r, r);
		}
		return dst;
	}

	private static Mat getAlignRotateMat(Mat src, Mat dst) {
		double a = 0, b = 0, d = 0;
		for (int i = 0; i < src.rows() / 2; i++) {
			double sx = src.get(i * 2, 0)[0];
			double sy = src.get(i * 2 + 1, 0)[0];
			double dx = dst.get(i * 2, 0)[0];
			double dy = dst.get(i * 2 + 1, 0)[0];
			d += sx * sx + sy * sy;
			a += sx * dx + sy * dy;
			b += sx * dy - sy * dx;
		}
		a /= d;
		b /= d;
		Mat result = new Mat(2, 2, CvType.CV_32F);
		result.put(0, 0, a, -b, b, a);
		return result;
	}

}

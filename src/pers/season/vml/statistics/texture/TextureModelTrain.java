package pers.season.vml.statistics.texture;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public class TextureModelTrain {

	public static void visualize(TextureModel tm) {
		JFrame win = new JFrame();
		TextureInstance textureModel = new TextureInstance(tm);
		for (int feature = 0; feature < tm.Z_SIZE; feature++) {
			win.setTitle("Feature = " + feature);
			double[] seq = new double[] { 0, 3, -3, 0 };
			for (int s = 0; s < seq.length - 1; s++) {
				for (double i = seq[s]; Math.abs(i - seq[s + 1]) > 0.001; i += 0.1 * Math.signum(seq[s + 1] - seq[s])) {
					textureModel.Z.put(feature, 0, tm.e.get(feature, 0)[0] * i);
					Mat canvas = Mat.zeros(tm.resolutionY, tm.resolutionX, CvType.CV_32F);
					textureModel.printTo(canvas);
					ImUtils.imshow(win, canvas, 5);
					System.gc();

				}
			}
		}
	}

	public static void train(String outputDir, double fractionRemain, int resolution_x, int resolution_y,
			boolean saveTransitionalData) {
		System.out.println("training texture model ...");

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

		Mat meanShape = new Mat();
		Core.gemm(shapes, Mat.ones(shapes.cols(), 1, CvType.CV_32F), 1, new Mat(), 0, meanShape);
		Core.divide(meanShape, new Scalar(shapes.cols()), meanShape);

		// normalize meanShape
		Mat meanShape_x = new Mat(meanShape.rows() / 2, 1, CvType.CV_32F);
		Mat meanShape_y = new Mat(meanShape.rows() / 2, 1, CvType.CV_32F);
		for (int i = 0; i < meanShape.rows() / 2; i++) {
			meanShape_x.put(i, 0, meanShape.get(i * 2, 0)[0]);
			meanShape_y.put(i, 0, meanShape.get(i * 2 + 1, 0)[0]);
		}
		Core.normalize(meanShape_x, meanShape_x, 1, resolution_x - 1, Core.NORM_MINMAX);
		Core.normalize(meanShape_y, meanShape_y, 1, resolution_y - 1, Core.NORM_MINMAX);
		for (int i = 0; i < meanShape.rows() / 2; i++) {
			meanShape.put(i * 2, 0, meanShape_x.get(i, 0)[0]);
			meanShape.put(i * 2 + 1, 0, meanShape_y.get(i, 0)[0]);
		}

		ImUtils.saveMat(meanShape, outputDir + "meanShape");
		System.out.println("normal face shape generated");

		// create triangle delaunay
		int[][] delaunay = TextureModel.createDelaunay(new Rect(0, 0, resolution_x, resolution_y), meanShape);
		Mat delaunayMat = new Mat(delaunay.length, 3, CvType.CV_32F);
		for (int i = 0; i < delaunay.length; i++) {
			for (int j = 0; j < 3; j++) {
				delaunayMat.put(i, j, delaunay[i][j]);
			}
		}
		ImUtils.saveMat(delaunayMat, outputDir + "delaunay");
		System.out.println("triangle delaunay created");
		System.out.println("triangleCounts : " + delaunay.length);
		if (saveTransitionalData)
			ImUtils.showDelaunay(meanShape, delaunay, resolution_x, resolution_y);

		// affine faces
		Mat X = new Mat();
		for (int i = 0; i < MuctData.getSize(); i++) {
			Mat pic = MuctData.getGrayJpg(i);
			pic.convertTo(pic, CvType.CV_32F);
			Mat normFace = Mat.zeros(resolution_y, resolution_x, CvType.CV_32F);
			TextureModel.AfflineTexture(pic, MuctData.getPtsMat(i), normFace, meanShape, delaunay);
			normFace = normFace.reshape(1, 1);
			X.push_back(normFace);
			if (i % 100 == 0 || i == MuctData.getSize() - 1)
				System.out.println("affining transforming face ... " + i + "/" + MuctData.getSize());
			System.gc();
		}
		X = X.t();
		if (saveTransitionalData) {
			System.out.println("saving to file ...");
			ImUtils.saveMatAsInt(X, outputDir + "normFaces");
		}
		System.out.println("affining transforming face completed");

		// print a sample. (clone makes matrix continuous, which created by
		// push_back is not continuous)
		if (saveTransitionalData)
			ImUtils.imshow(new JFrame(), X.col(0).clone().reshape(1, resolution_y), 3);

		// apply svd (strongly recommend using matlab-svds function to do this
		// process)
		Mat meanX = new Mat();
		Core.gemm(X, Mat.ones(X.cols(), 1, CvType.CV_32F), 1, new Mat(), 0, meanX);
		Core.divide(meanX, new Scalar(X.cols()), meanX);
		ImUtils.saveMat(meanX, outputDir + "X_mean");

		for (int i = 0; i < X.cols(); i++)
			Core.subtract(X.col(i), meanX, X.col(i));
		System.out.println("applying svd ...");
		Mat U = new Mat();
		Mat S = new Mat();
		Mat covar = new Mat();
		Core.gemm(X, X.t(), 1, new Mat(), 0, covar);
		Core.SVDecomp(covar, S, U, new Mat());
		if (saveTransitionalData) {
			ImUtils.saveMat(U, outputDir + "U_full");
			ImUtils.saveMat(S, outputDir + "S");
		}
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
		ImUtils.saveMat(U, outputDir + "U");
		System.out.println("svd completed");

		// calculate Z and mean/stddev
		System.out.println("calculating Z and mean/stddev ...");
		Mat Z = new Mat();
		Core.gemm(U.t(), X, 1, new Mat(), 0, Z);
		if (saveTransitionalData)
			ImUtils.saveMat(Z, outputDir + "Z");
		Mat z_mean = new Mat(Z.rows(), 1, CvType.CV_32F);
		Mat z_stddev = new Mat(Z.rows(), 1, CvType.CV_32F);
		for (int i = 0; i < Z.rows(); i++) {
			MatOfDouble tmean = new MatOfDouble();
			MatOfDouble tstddev = new MatOfDouble();
			Core.meanStdDev(Z.row(i), tmean, tstddev);
			z_mean.put(i, 0, tmean.get(0, 0)[0]);
			z_stddev.put(i, 0, tstddev.get(0, 0)[0]);
		}
		ImUtils.saveMat(z_stddev, outputDir + "Z_e");
		if (saveTransitionalData)
			ImUtils.saveMat(z_mean, outputDir + "Z_mean");

		System.out.println("done!");

	}

}

package pers.season.vml.statistics.appearance;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Scalar;
import pers.season.vml.statistics.shape.ShapeInstance;
import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.statistics.texture.TextureInstance;
import pers.season.vml.statistics.texture.TextureModel;
import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public class AppearanceModelTrain {

	public static void visualize(AppearanceModel am) {
		JFrame win = new JFrame();
		ShapeInstance shape = new ShapeInstance(am.sm);
		shape.setFromParams(300, 0, 250, 250);
		TextureInstance texture = new TextureInstance(am.tm);

		AppearanceInstance app = new AppearanceInstance(am);
		app.setFromModels(shape.getZ(), texture.getZ());
		for (int feature = 4; feature < am.Z_SIZE; feature++) {
			win.setTitle("Feature = " + feature);
			double[] seq = new double[] { 0, 3, -3, 0 };
			for (int s = 0; s < seq.length - 1; s++) {
				for (double i = seq[s]; Math.abs(i - seq[s + 1]) > 0.001; i += 0.25
						* Math.signum(seq[s + 1] - seq[s])) {
					app.Z.put(feature, 0, am.e.get(feature, 0)[0] * i);
					Mat canvas = Mat.zeros(500, 500, CvType.CV_32F);
					app.printTo(canvas, true);
					ImUtils.imshow(win, canvas, 1);
					System.gc();

				}
			}

		}
	}

	public static void train(ShapeModel sm, TextureModel tm, String outputDir, double shapeWeightRatio,
			double fractionRemain, boolean saveTransitionalData) {
		System.out.println("training appearnce model ...");

		// calculate shape/texture project(Z)
		System.out.println("calculating shape/texutre Z ... ");
		Mat textureZ = new Mat();
		Mat shapeZ = new Mat();
		for (int i = 0; i < MuctData.getSize(); i++) {
			Mat pic = MuctData.getGrayJpg(i);
			pic.convertTo(pic, CvType.CV_32F);
			shapeZ.push_back(sm.getZe4fromX(MuctData.getPtsMat(i)).t());
			textureZ.push_back(tm.getZfromX(tm.getNormFace(pic, MuctData.getPtsMat(i))).t());
			if (i % 100 == 0 || i == MuctData.getSize() - 1)
				System.out.println(i + "/" + MuctData.getSize());
			System.gc();
		}
		shapeZ = shapeZ.t();
		textureZ = textureZ.t();
		if (saveTransitionalData) {
			System.out.println("saving shape/texutre Z to file ... ");
			ImUtils.saveMat(textureZ, outputDir + "textureZ");
			ImUtils.saveMat(shapeZ, outputDir + "shapeZ");
		}

		// evaluate shape weight
		double shapeWeight = shapeWeightRatio * Core.norm(textureZ) / Core.norm(shapeZ);
		Mat shapeWeightMat = new Mat(1, 1, CvType.CV_32F);
		shapeWeightMat.put(0, 0, shapeWeight);
		ImUtils.saveMat(shapeWeightMat, outputDir + "shapeWeight");
		System.out.println("suggested shape weight = " + shapeWeight);
		Core.multiply(shapeZ, new Scalar(shapeWeight), shapeZ);

		// apply svd to shape+texture vector
		System.out.println("generating appearance vector ...");
		Mat appX = new Mat();
		appX.push_back(shapeZ);
		appX.push_back(textureZ);
		if (saveTransitionalData)
			ImUtils.saveMat(appX, outputDir + "app_X");

		System.out.println("applying svd ...");
		Mat U = new Mat();
		Mat S = new Mat();
		Mat covar = new Mat();
		Core.gemm(appX, appX.t(), 1, new Mat(), 0, covar);
		Core.SVDecomp(covar, S, U, new Mat());
		ImUtils.saveMat(S, outputDir + "S");
		if (saveTransitionalData) {
			ImUtils.saveMat(U, outputDir + "U_full");
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
		Mat appZ = new Mat();
		Core.gemm(U.t(), appX, 1, new Mat(), 0, appZ);
		ImUtils.saveMat(appZ, outputDir + "Z");
		Mat z_mean = new Mat(appZ.rows(), 1, CvType.CV_32F);
		Mat z_stddev = new Mat(appZ.rows(), 1, CvType.CV_32F);
		for (int i = 0; i < appZ.rows(); i++) {
			MatOfDouble tmean = new MatOfDouble();
			MatOfDouble tstddev = new MatOfDouble();
			Core.meanStdDev(appZ.row(i), tmean, tstddev);
			z_mean.put(i, 0, tmean.get(0, 0)[0]);
			z_stddev.put(i, 0, tstddev.get(0, 0)[0]);
		}
		Mat z_mean_p4 = Mat.zeros(z_mean.rows() + 4, z_mean.cols(), CvType.CV_32F);
		Mat z_stddev_p4 = Mat.zeros(z_stddev.rows() + 4, z_stddev.cols(), CvType.CV_32F);
		z_mean_p4.put(0, 0, new float[] { -1, -1, -1, -1 });
		z_stddev_p4.put(0, 0, new float[] { -1, -1, -1, -1 });
		z_mean.copyTo(z_mean_p4.rowRange(4, z_mean_p4.rows()));
		z_stddev.copyTo(z_stddev_p4.rowRange(4, z_stddev_p4.rows()));
		ImUtils.saveMat(z_stddev, outputDir + "Z_e");
		if (saveTransitionalData)
			ImUtils.saveMat(z_mean, outputDir + "Z_mean");

		System.out.println("done!");

	}

}

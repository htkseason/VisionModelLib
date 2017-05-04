package pers.season.vml.test;

import java.io.IOException;
import java.util.UUID;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.statistics.appearance.AppearanceFitting;
import pers.season.vml.statistics.appearance.AppearanceModel;
import pers.season.vml.statistics.appearance.AppearanceModelTrain;
import pers.season.vml.statistics.regressor.LearningParams;
import pers.season.vml.statistics.regressor.RegressorTrain;
import pers.season.vml.statistics.shape.ShapeInstance;
import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.statistics.shape.ShapeModelTrain;
import pers.season.vml.statistics.texture.TextureInstance;
import pers.season.vml.statistics.texture.TextureModel;
import pers.season.vml.statistics.texture.TextureModelTrain;
import pers.season.vml.util.*;

public final class Entrance {

	static {
		// todo: x64/x86 judge
		// System.loadLibrary("lib/opencv_java2413_x64");
		System.loadLibrary("lib/opencv_java320_x64");
	}

	public static void main(String[] args) throws IOException {
		MuctData.init("e:/muct/jpg", "e:/muct/muct76-opencv.csv", MuctData.no_ignore);

		Mat refShape = RegressorTrain.getRefShape(100, 100);
		int point = 0;
		LearningParams lp = new LearningParams();
		lp.iteration = 1000;
		
		Size patchSize = new Size(61, 61);

		Mat theta = RegressorTrain.trainLinearModel(refShape, point, patchSize, new Size(61, 61), 3, 0.2, lp);
		RegressorTrain.measureDistribution("e:/Cool/OneDrive/temp/", "maxloc", "theta", refShape, point, theta,
				patchSize, new Size(61, 61), 0.2, false);

		// ShapeModelTrain.train("models/shape/", 0.90, false);
		// ShapeModel sm = ShapeModel.load("models/shape/", "V", "Z_e");
		// ShapeModelTrain.visualize(sm);

		// TextureModelTrain.train("models/texture/", 0.95, 100, 100, true);
		// TextureModel tm = TextureModel.load("models/texture/", "U", "X_mean",
		// "Z_e", "meanShape", "delaunay");
		// TextureModelTrain.visualize(tm);

		// AppearanceModelTrain.train(sm, tm, "models/appearance/", 1.5, 0.98,
		// false);
		// AppearanceModel am = AppearanceModel.load(sm, tm,
		// "models/appearance/", "U", "Z_e", "shapeWeight");
		// AppearanceModelTrain.visualize(am);

		System.out.println("program ended.");

	}

}

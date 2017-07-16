package pers.season.vml.test;

import java.io.IOException;
import java.util.UUID;

import javax.swing.JFrame;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.AKAZE;
import org.opencv.features2d.Feature2D;
import org.opencv.features2d.KAZE;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.videoio.VideoCapture;

import pers.season.vml.ar.CameraData;
import pers.season.vml.ar.Engine3D;
import pers.season.vml.ar.FeatureTracker;
import pers.season.vml.ar.MotionFilter;
import pers.season.vml.ml.LearningParams;
import pers.season.vml.statistics.appearance.AppearanceFitting;
import pers.season.vml.statistics.appearance.AppearanceModel;
import pers.season.vml.statistics.appearance.AppearanceModelTrain;
import pers.season.vml.statistics.regressor.RegressorSet;
import pers.season.vml.statistics.regressor.RegressorTrain;
import pers.season.vml.statistics.sdm.SdmHogDescriptor;
import pers.season.vml.statistics.sdm.SdmModel;
import pers.season.vml.statistics.sdm.SdmTrain;
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
		visionModelDemo();

	}

	public static void visionModelDemo() {
		 MuctData.init("e:/muct/jpg", "e:/muct/muct76-opencv.csv",
		 MuctData.no_ignore);

		ShapeModel sm = ShapeModel.load("models/shape/", "V", "Z_e");
		// ShapeModelTrain.visualize(sm,8);

		// TextureModelTrain.train("models/texturex/", 0.98, 20, 30, false);
		TextureModel tm = TextureModel.load("models/texture/", "U", "X_mean", "Z_e", "meanShape", "delaunay");
		// TextureModelTrain.visualize(tm,8);

		// AppearanceModelTrain.train(sm, tm, "models/appearancex/", 1.5, 0.98, false);
		AppearanceModel am = AppearanceModel.load(sm, tm, "models/appearance/", "U", "Z_e", "shapeWeight");
		// AppearanceModelTrain.visualize(am);

		FaceDetector fd = FaceDetector.load("models/haarcascade_frontalface_default.xml");
		RegressorSet rs = RegressorSet.load("models/regressor/", "patch_76_size61", "refShape", new Size(61, 61));
		
		SdmTrain.train(rs.refShape, new Size(32,32));
	}
}

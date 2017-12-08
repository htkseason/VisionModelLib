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

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import pers.season.vml.ar.CameraData;
import pers.season.vml.ar.Engine3D;
import pers.season.vml.ar.MarkerDetector;
import pers.season.vml.ar.TemplateDetector;
import pers.season.vml.ar.MotionFilter;
import pers.season.vml.ml.LearningParams;
import pers.season.vml.statistics.appearance.AppearanceFitting;
import pers.season.vml.statistics.appearance.AppearanceModel;
import pers.season.vml.statistics.appearance.AppearanceModelTrain;
import pers.season.vml.statistics.patch.PatchSet;
import pers.season.vml.statistics.patch.PatchTrain;
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
		//visionModelDemo();

		MarkerDetector md = new MarkerDetector();
		md.findMarkers(Imgcodecs.imread("e:/test.png", Imgcodecs.IMREAD_GRAYSCALE));
		
	}

	public static void visionModelDemo() {
		MuctData.init("e:/muct/jpg", "e:/muct/muct76-opencv.csv", MuctData.no_ignore);

		ShapeModel sm = ShapeModel.load("models/shape/", "V", "Z_e");
		// ShapeModelTrain.visualize(sm,8);

		// TextureModelTrain.train("models/texturex/", 0.98, 20, 30, false);
		TextureModel tm = TextureModel.load("models/texture/", "U", "X_mean", "Z_e", "meanShape", "delaunay");
		// TextureModelTrain.visualize(tm,8);

		// AppearanceModelTrain.train(sm, tm, "models/appearancex/", 1.5, 0.98, false);
		AppearanceModel am = AppearanceModel.load(sm, tm, "models/appearance/", "U", "Z_e", "shapeWeight");
		// AppearanceModelTrain.visualize(am);

		FaceDetector fd = FaceDetector.load("models/lbpcascade_frontalface.xml");
		PatchSet ps = PatchSet.load("models/patch/", "patch_76_61x61", "refShape", new Size(61, 61));

		SdmTrain.train(ps.refShape, new Size(32,32));
		SdmModel sdm = SdmModel.load("models/sdm_500_0sample/");
		VideoCapture vc = new VideoCapture();
		vc.open(0);
		JFrame win = new JFrame();
		while (true) {
			Mat pic = new Mat();
			vc.read(pic);
			pic = MuctData.getGrayJpg(555);
			Rect[] faceRects = fd.searchFace(pic);
			if (faceRects.length == 0)
				continue;
			Rect faceRect = faceRects[0];
			ShapeInstance shape = new ShapeInstance(sm);

			shape.setFromParams(faceRect.width * 0.9, 0, faceRect.x + faceRect.width / 2,
					faceRect.y + faceRect.height / 2 + faceRect.height * 0.12);
			Mat pts = shape.getX();
			while (true) {
				vc.read(pic);
				pic = MuctData.getGrayJpg(555);
				pts=shape.getX();
				for (int iter = 0; iter <10; iter++) {

					Mat R = PatchSet.getPtsAffineTrans(pts, ps.refShape, pic.width() / 2, pic.height() / 2);
					Mat feature = SdmModel.computeFeature(pic, pts, ps.refShape, new Size(32, 32));
					Mat residual = SdmModel.calcResidual(feature, sdm.theta);

					//System.out.println(Core.norm(residual));

					Core.multiply(residual, new Scalar(0.5), residual);
					Mat ptsAff = PatchSet.warpPtsAffine(pts, R);
					Core.add(ptsAff, residual, ptsAff);
					pts = PatchSet.reversePtsAffine(ptsAff, R);
				}

				Mat vpic = pic.clone();
				for (int i = 0; i < pts.rows() / 2; i++) {
					Imgproc.circle(vpic, new Point(pts.get(i * 2, 0)[0], pts.get(i * 2 + 1, 0)[0]), 2, new Scalar(255),
							2);
				}
				ImUtils.imshow(win, vpic, 1);
			}
		}

	}
}

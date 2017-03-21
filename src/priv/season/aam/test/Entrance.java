package priv.season.aam.test;

import java.io.IOException;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import priv.season.aam.statistics.appearance.AppearanceFitting;
import priv.season.aam.statistics.appearance.AppearanceModel;
import priv.season.aam.statistics.appearance.AppearanceModelTrain;
import priv.season.aam.statistics.regressor.RegressorTrain;
import priv.season.aam.statistics.shape.ShapeInstance;
import priv.season.aam.statistics.shape.ShapeModel;
import priv.season.aam.statistics.shape.ShapeModelTrain;
import priv.season.aam.statistics.texture.TextureInstance;
import priv.season.aam.statistics.texture.TextureModel;
import priv.season.aam.statistics.texture.TextureModelTrain;
import priv.season.aam.util.CwMat;
import priv.season.aam.util.FaceDetector;
import priv.season.aam.util.MuctData;
import priv.season.aam.util.ImUtils;

public final class Entrance {

	static {
		// todo: x64/x86 judge
	//	System.loadLibrary("lib/opencv_java2413_x64");
		System.loadLibrary("lib/opencv_java320_x64");
	}

	public static void main(String[] args) {
		warpFace();
		//aamFittingDemo();
		//RegressorTrain.trainLR();
		System.out.println("program ended.");
	}
	
	public static void warpFace () {
		TextureModel.init("e:/Cool/OneDrive/aam/texture/", "U", "X_mean", "Z_e", "meanShape", "delaunay");
		Mat pic = MuctData.getGrayJpg(1);
		Mat srcpts = MuctData.getPtsMat(1);
		Mat addPts = new Mat(8*2,1,CvType.CV_64F);
		addPts.put(0, 0, 100,100,400,100,100,600,400,600, 200,100, 200,600,100,300,400,300);
		srcpts.push_back(addPts);
		Mat dstpts = MuctData.getPtsMat(1000);
		dstpts.push_back(addPts);
		int[][] delaunay = TextureModel.createDelaunay(new Rect(0,0,480,640),srcpts);
		System.out.println("triangleCounts : " + delaunay.length);
		Mat delaunay_show = Mat.zeros(640, 480, CvType.CV_64F);
		for (int i = 0; i < delaunay.length; i++) {
			int p1c = delaunay[i][0];
			int p2c = delaunay[i][1];
			int p3c = delaunay[i][2];
			double p1x = srcpts.get(p1c * 2, 0)[0];
			double p2x = srcpts.get(p2c * 2, 0)[0];
			double p3x = srcpts.get(p3c * 2, 0)[0];
			double p1y = srcpts.get(p1c * 2 + 1, 0)[0];
			double p2y = srcpts.get(p2c * 2 + 1, 0)[0];
			double p3y = srcpts.get(p3c * 2 + 1, 0)[0];
			Imgproc.line(delaunay_show, new Point(p1x, p1y), new Point(p2x, p2y), new Scalar(255));
			Imgproc.line(delaunay_show, new Point(p3x, p3y), new Point(p2x, p2y), new Scalar(255));
			Imgproc.line(delaunay_show, new Point(p1x, p1y), new Point(p3x, p3y), new Scalar(255));
		}
		ImUtils.imshow(delaunay_show);
		Mat dstpic = Mat.zeros(pic.size(),pic.type());
		ImUtils.imshow(pic);
		TextureModel.AfflineTexture(pic, srcpts, dstpic, dstpts, delaunay);
		ImUtils.imshow(dstpic);
	}

	public static void aamFittingDemo() {
		// ShapeModelTrain.train("e:/Cool/OneDrive/aam/shape/", 0.90, false);
		ShapeModel.init("E:/Cool/OneDrive/aam/shape/", "V", "Z_e");
		// ShapeModelTrain.visualize();

		// TextureModelTrain.train("e:/Cool/OneDrive/aam/texture/", 0.95, 100,
		// 100, true);
		TextureModel.init("e:/Cool/OneDrive/aam/texture/", "U", "X_mean", "Z_e", "meanShape", "delaunay");
		// TextureModelTrain.visualize();

		// AppearanceModelTrain.train("e:/Cool/OneDrive/aam/appearance/", 0.98,
		// false);
		AppearanceModel.init("e:/Cool/OneDrive/aam/appearance/", "U", "Z_e", "shapeWeight");
		// AppearanceModelTrain.visualize();

		ImUtils.startTiming();
		Mat pic = MuctData.getGrayJpg(555);
		// Mat pic = Utils.loadImage("e:/t2.jpg");


		Rect faceRect = FaceDetector.searchFace(pic);

		ImUtils.imshow(pic);
		pic.convertTo(pic, CvType.CV_64F);
		Mat v_pic = new Mat(pic.size(), pic.type());

		TextureInstance texture = new TextureInstance();
		ShapeInstance shape = new ShapeInstance(faceRect.width * 0.9, 0, faceRect.x + faceRect.width / 2,
				faceRect.y + faceRect.height / 2 + faceRect.height * 0.15);

		AppearanceFitting app = new AppearanceFitting(pic, shape.getZ(), texture.getZ());

		pic.copyTo(v_pic);
		app.printTo(v_pic);
		System.out.println(ImUtils.getCostE(app.getCost()) + "\t\t" + ImUtils.getTiming() + " ms");
		double preCost = Double.MAX_VALUE;
		JFrame win = new JFrame();
		for (int iter = 0; iter < 1000; iter++) {

			ImUtils.imshow(win, v_pic, 1);

			ImUtils.startTiming();
			System.gc();
			Mat gra = app.getGradient();

			Core.multiply(gra, new Scalar(0.1), gra);
			int times = 0;
			while (true) {
				times++;
				app.updata(gra);
				double cost = ImUtils.getCostE(app.getCost());
				if (preCost < cost) {
					Core.multiply(gra, new Scalar(-1), gra);
					app.updata(gra);
					break;
				}
				preCost = cost;
			}
			// app.clamp(3);
			pic.copyTo(v_pic);
			app.printTo(v_pic);
			System.out.println(times + ", " + ImUtils.getCostE(app.getCost()) + "\t\t" + ImUtils.getTiming() + " ms");

		}

		System.out.println("\ndone!");
		
	}

}

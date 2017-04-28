package pers.season.vml.statistics.texture;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

import javax.rmi.CORBA.Util;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat6;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Subdiv2D;

import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;
import pers.season.vml.util.Triangle;

public class TextureModel {

	public int delaunay[][];
	public Mat stdShape;
	public Mat meanX;
	public Mat U, S, e;
	public int X_SIZE, Z_SIZE;
	public int resolutionX, resolutionY;
	protected static final int CORE_COUNTS = Runtime.getRuntime().availableProcessors();
	protected static ExecutorService threadPool = Executors.newCachedThreadPool();

	protected TextureModel() {
		
	}
	
	public static TextureModel load(String dataPath, String U_name, String meanX_name, String e_name,
			String meanShape_name, String delaunay_name) {
		TextureModel tm = new TextureModel();
		tm.U = ImUtils.loadMat(dataPath + U_name);
		tm.meanX = ImUtils.loadMat(dataPath + meanX_name);
		tm.e = ImUtils.loadMat(dataPath + e_name);

		tm.stdShape = ImUtils.loadMat(dataPath + meanShape_name);
		Mat delaunayMat = ImUtils.loadMat(dataPath + delaunay_name);
		tm.delaunay = new int[delaunayMat.rows()][];
		for (int i = 0; i < delaunayMat.rows(); i++) {
			tm.delaunay[i] = new int[] { (int) delaunayMat.get(i, 0)[0], (int) delaunayMat.get(i, 1)[0],
					(int) delaunayMat.get(i, 2)[0] };

		}

		tm.X_SIZE = tm.U.rows();
		tm.Z_SIZE = tm.U.cols();

		Mat stdShapeX = new Mat(tm.stdShape.rows() / 2, 1, CvType.CV_32F);
		Mat stdShapeY = new Mat(tm.stdShape.rows() / 2, 1, CvType.CV_32F);
		for (int i = 0; i < tm.stdShape.rows() / 2; i++) {
			stdShapeX.put(i, 0, tm.stdShape.get(i * 2, 0)[0]);
			stdShapeY.put(i, 0, tm.stdShape.get(i * 2 + 1, 0)[0]);
		}
		MinMaxLocResult mmX = Core.minMaxLoc(stdShapeX);
		MinMaxLocResult mmY = Core.minMaxLoc(stdShapeY);
		tm.resolutionX = (int) Math.round(mmX.maxVal - mmX.minVal + 2);
		tm.resolutionY = (int) Math.round(mmY.maxVal - mmY.minVal + 2);

		System.out.println("TextureModel inited. " + tm.X_SIZE + " --> " + tm.Z_SIZE);
		return tm;

	}


	public void printTo(Mat Z, Mat dst, Mat shape) {
		Mat X = getXfromZ(Z).reshape(1, resolutionY);
		AfflineTexture(X, stdShape, dst, shape, delaunay);
	}

	public Mat getZfromX(Mat X) {
		Mat result = new Mat();
		X = X.reshape(1, X_SIZE);
		Core.subtract(X, meanX, X);
		Core.gemm(U.t(), X, 1, new Mat(), 0, result);
		return result;
	}

	public Mat getXfromZ(Mat Z) {
		Mat X = new Mat();
		Core.gemm(U, Z, 1, new Mat(), 0, X);
		Core.add(X, meanX, X);
		return X;
	}

	public static int[][] createDelaunay(Rect roi, Mat pts) {
		DecimalFormat df = new DecimalFormat("#.0000");
		List<int[]> resultLst = new LinkedList<int[]>();
		ArrayList<String> pointIndexSearchLst = new ArrayList<String>();
		Subdiv2D sd = new Subdiv2D();

		sd.initDelaunay(roi);
		for (int i = 0; i < pts.rows() / 2; i++) {
			sd.insert(new Point(pts.get(i * 2, 0)[0], pts.get(i * 2 + 1, 0)[0]));
			pointIndexSearchLst.add(df.format(pts.get(i * 2, 0)[0]) + "," + df.format(pts.get(i * 2 + 1, 0)[0]));
		}
		MatOfFloat6 mof6 = new MatOfFloat6();
		sd.getTriangleList(mof6);
		for (int i = 0; i < mof6.rows(); i++) {
			double[] data = new double[6];
			data = mof6.get(i, 0);

			Point p1 = new Point(data[0], data[1]);
			Point p2 = new Point(data[2], data[3]);
			Point p3 = new Point(data[4], data[5]);
			if (!p1.inside(roi) || !p2.inside(roi) || !p3.inside(roi)) {
				continue;
			}
			int i1 = pointIndexSearchLst.indexOf(df.format(data[0]) + "," + df.format(data[1]));
			int i2 = pointIndexSearchLst.indexOf(df.format(data[2]) + "," + df.format(data[3]));
			int i3 = pointIndexSearchLst.indexOf(df.format(data[4]) + "," + df.format(data[5]));
			resultLst.add(new int[] { i1, i2, i3 });
		}

		int[][] result = new int[resultLst.size()][];
		resultLst.toArray(result);
		return result;
	}

	public static void AfflineTexture(Mat srcpic, Mat srcpts, Mat dstpic, Mat dstpts, int[][] delaunay) {
		Semaphore sema = new Semaphore(0);

		for (int threadIndex = 0; threadIndex < CORE_COUNTS; threadIndex++) {
			final int curThreadIndex = threadIndex;
			threadPool.execute(new Runnable() {
				@Override
				public void run() {
					for (int i = 0; i < delaunay.length; i++) {
						if (curThreadIndex != i % CORE_COUNTS)
							continue;
						int x1i = delaunay[i][0] * 2;
						int y1i = delaunay[i][0] * 2 + 1;
						int x2i = delaunay[i][1] * 2;
						int y2i = delaunay[i][1] * 2 + 1;
						int x3i = delaunay[i][2] * 2;
						int y3i = delaunay[i][2] * 2 + 1;
						double x1 = srcpts.get(x1i, 0)[0];
						double y1 = srcpts.get(y1i, 0)[0];
						double x2 = srcpts.get(x2i, 0)[0];
						double y2 = srcpts.get(y2i, 0)[0];
						double x3 = srcpts.get(x3i, 0)[0];
						double y3 = srcpts.get(y3i, 0)[0];

						double nx1 = dstpts.get(x1i, 0)[0];
						double ny1 = dstpts.get(y1i, 0)[0];
						double nx2 = dstpts.get(x2i, 0)[0];
						double ny2 = dstpts.get(y2i, 0)[0];
						double nx3 = dstpts.get(x3i, 0)[0];
						double ny3 = dstpts.get(y3i, 0)[0];

						Triangle t = new Triangle(srcpic, x1, y1, x2, y2, x3, y3);
						t.shift(nx1, ny1, nx2, ny2, nx3, ny3);

						t.transTextureTo(dstpic);

					}
					sema.release();
				}
			});
		}
		try {
			sema.acquire(CORE_COUNTS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}

	public Mat getNormFace(Mat pic, Mat pts) {

		float[] ptsdata = new float[pts.rows()];
		pts.get(0, 0, ptsdata);

		Mat result = Mat.zeros(new Size(resolutionX, resolutionY), pic.type());

		AfflineTexture(pic, pts, result, stdShape, delaunay);

		return result;

	}

}

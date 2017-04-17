package pers.season.vml.util;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Subdiv2D;

public class Triangle {
	public double x1, x2, x3, y1, y2, y3;
	public double nx1, nx2, nx3, ny1, ny2, ny3;
	public Mat texture;

	public Triangle(Mat texture, double x1, double y1, double x2, double y2, double x3, double y3) {
		int offset = 10;
		int minX = (int) Math.min(Math.min(x1, x2), x3) - offset;
		minX = minX < 0 ? 0 : minX;
		int minY = (int) Math.min(Math.min(y1, y2), y3) - offset;
		minY = minY < 0 ? 0 : minY;
		int maxX = (int) Math.max(Math.max(x1, x2), x3) + offset;
		maxX = maxX > texture.cols() ? texture.cols() : maxX;
		int maxY = (int) Math.max(Math.max(y1, y2), y3) + offset;
		maxY = maxY > texture.rows() ? texture.rows() : maxY;

		this.x1 = x1;
		this.x2 = x2;
		this.x3 = x3;
		this.y1 = y1;
		this.y2 = y2;
		this.y3 = y3;
		// TODO : optimize
		this.texture = texture;// .submat(new Rect(minX, minY, maxX - minX, maxY
								// - minY));
		shift(this.x1, this.y1, this.x2, this.y2, this.x3, this.y3);
	}

	public void shift(double nx1, double ny1, double nx2, double ny2, double nx3, double ny3) {
		this.nx1 = nx1;
		this.nx2 = nx2;
		this.nx3 = nx3;
		this.ny1 = ny1;
		this.ny2 = ny2;
		this.ny3 = ny3;
	}

	public void transTextureTo(Mat dst) {
		// calculate translation matrix
		MatOfPoint2f srcPosition = new MatOfPoint2f(new Point(x1, y1), new Point(x2, y2), new Point(x3, y3));
		MatOfPoint2f dstPosition = new MatOfPoint2f(new Point(nx1, ny1), new Point(nx2, ny2), new Point(nx3, ny3));
		Mat trans = Imgproc.getAffineTransform(srcPosition, dstPosition);

		// create triangle mask
		Mat mask = Mat.zeros(dst.size(), CvType.CV_8U);
		MatOfPoint temp_mop = new MatOfPoint(new Point(nx1, ny1), new Point(nx2, ny2), new Point(nx3, ny3));
		Imgproc.fillConvexPoly(mask, temp_mop, new Scalar(255, 255, 255));
		// copy to dst
		Mat tempdst = new Mat();
		Imgproc.warpAffine(texture, tempdst, trans, dst.size());
		synchronized (dst) {
			tempdst.copyTo(dst, mask);
		}

	}

}

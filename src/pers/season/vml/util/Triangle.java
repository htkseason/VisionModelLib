package pers.season.vml.util;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class Triangle {
	public double x1, x2, x3, y1, y2, y3;
	public double nx1, nx2, nx3, ny1, ny2, ny3;
	public Mat texture;

	public Triangle(Mat texture, double x1, double y1, double x2, double y2, double x3, double y3) {
		this.x1 = x1;
		this.x2 = x2;
		this.x3 = x3;
		this.y1 = y1;
		this.y2 = y2;
		this.y3 = y3;

		this.texture = texture;
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
		int offset = 0;
		int minX = (int) Math.min(Math.min(nx1, nx2), nx3) - offset;
		minX = minX < 0 ? 0 : minX;
		int minY = (int) Math.min(Math.min(ny1, ny2), ny3) - offset;
		minY = minY < 0 ? 0 : minY;
		int maxX = (int) Math.max(Math.max(nx1, nx2), nx3) + offset;
		maxX = maxX >= dst.width() ? dst.width() - 1 : maxX;
		int maxY = (int) Math.max(Math.max(ny1, ny2), ny3) + offset;
		maxY = maxY >= dst.height() ? dst.height() - 1 : maxY;
		if (minY >= maxY || minX >= maxX)
			return;

		// calculate translation matrix
		MatOfPoint2f srcPosition = new MatOfPoint2f(new Point(x1, y1), new Point(x2, y2), new Point(x3, y3));
		MatOfPoint2f dstPosition = new MatOfPoint2f(new Point(nx1 - minX, ny1 - minY),
				new Point(nx2 - minX, ny2 - minY), new Point(nx3 - minX, ny3 - minY));
		Rect dstRoi = new Rect(minX, minY, maxX - minX, maxY - minY);
		Mat trans = Imgproc.getAffineTransform(srcPosition, dstPosition);

		// create triangle mask
		Mat mask = Mat.zeros(dstRoi.size(), CvType.CV_8U);
		MatOfPoint temp_mop = new MatOfPoint(new Point(nx1 - minX, ny1 - minY), new Point(nx2 - minX, ny2 - minY),
				new Point(nx3 - minX, ny3 - minY));
		Imgproc.fillConvexPoly(mask, temp_mop, new Scalar(255, 255, 255));
		// copy to dst
		Mat tempdst = new Mat();
		Imgproc.warpAffine(texture, tempdst, trans, new Size(maxX - minX, maxY - minY));
		synchronized (dst) {
			tempdst.copyTo(dst.submat(dstRoi), mask);
		}

	}

}

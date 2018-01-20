package pers.season.vml.ar;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.util.ImUtils;

public final class MyMarker {
	public final static int WIDTH = 7;
	public final static int HEIGHT = 7;
	public int code1 = 0;
	public int code2 = 0;

	public MyMarker() {

	}

	public static MyMarker parse(Mat pic, MatOfPoint2f refMarkerPts) {

		double threshold = 0.5;
		int lengthPerBlock = 10;
		int width = (int) WIDTH * lengthPerBlock;
		int height = (int) HEIGHT * lengthPerBlock;
		Mat markerPic = new Mat();
		Mat srcPts = new Mat(4, 1, CvType.CV_32FC2);
		srcPts.put(0, 0, 0, 0, width, 0, width, height, 0, height);
		Mat homo = Imgproc.getPerspectiveTransform(srcPts, refMarkerPts);
		Imgproc.warpPerspective(pic, markerPic, homo, new Size(width, height),
				Imgproc.WARP_INVERSE_MAP | Imgproc.INTER_CUBIC);
		Imgproc.threshold(markerPic, markerPic, 0, 255, Imgproc.THRESH_OTSU);

		Mat markerData = new Mat(HEIGHT, WIDTH, CvType.CV_8U);
		for (int y = 0; y < HEIGHT; y++)
			for (int x = 0; x < WIDTH; x++) {
				int nonZeroPixels = Core.countNonZero(markerPic.submat(y * lengthPerBlock, (y + 1) * lengthPerBlock,
						x * lengthPerBlock, (x + 1) * lengthPerBlock));
				markerData.put(y, x, nonZeroPixels > lengthPerBlock * lengthPerBlock * threshold ? 255 : 0);
			}

		if (markerData.width() != WIDTH || markerData.height() != HEIGHT)
			return null;

		MyMarker ret = new MyMarker();
		// check surrounding blocks
		if (Core.countNonZero(markerData.row(0)) != 0 || Core.countNonZero(markerData.row(HEIGHT - 1)) != 0
				|| Core.countNonZero(markerData.col(0)) != 0 || Core.countNonZero(markerData.col(WIDTH - 1)) != 0)
			return null;

		// check center
		if (markerData.get(HEIGHT / 2, WIDTH / 2)[0] == 0)
			return null;

		// check corner and adjust rotation
		if ((markerData.get(1, 1)[0] != 0 ? 1 : 0) + (markerData.get(1, WIDTH - 2)[0] != 0 ? 1 : 0)
				+ (markerData.get(HEIGHT - 2, WIDTH - 2)[0] != 0 ? 1 : 0)
				+ (markerData.get(HEIGHT - 2, 1)[0] != 0 ? 1 : 0) != 1)
			return null;

		if (markerData.get(1, WIDTH - 2)[0] != 0) {
			Core.rotate(markerData, markerData, Core.ROTATE_90_COUNTERCLOCKWISE);
			for (int p = 0; p < 3; p++) {
				Core.bitwise_xor(refMarkerPts.row(p), refMarkerPts.row((p + 1) % 4), refMarkerPts.row(p));
				Core.bitwise_xor(refMarkerPts.row(p), refMarkerPts.row((p + 1) % 4), refMarkerPts.row((p + 1) % 4));
				Core.bitwise_xor(refMarkerPts.row(p), refMarkerPts.row((p + 1) % 4), refMarkerPts.row(p));
			}
		} else if (markerData.get(HEIGHT - 2, WIDTH - 2)[0] != 0) {
			Core.rotate(markerData, markerData, Core.ROTATE_180);
			for (int p = 0; p < 2; p++) {
				Core.bitwise_xor(refMarkerPts.row(p), refMarkerPts.row((p + 2) % 4), refMarkerPts.row(p));
				Core.bitwise_xor(refMarkerPts.row(p), refMarkerPts.row((p + 2) % 4), refMarkerPts.row((p + 2) % 4));
				Core.bitwise_xor(refMarkerPts.row(p), refMarkerPts.row((p + 2) % 4), refMarkerPts.row(p));
			}
		} else if (markerData.get(HEIGHT - 2, 1)[0] != 0) {
			Core.rotate(markerData, markerData, Core.ROTATE_90_CLOCKWISE);
			for (int p = 3; p > 0; p--) {
				Core.bitwise_xor(refMarkerPts.row(p), refMarkerPts.row((p + 1) % 4), refMarkerPts.row(p));
				Core.bitwise_xor(refMarkerPts.row(p), refMarkerPts.row((p + 1) % 4), refMarkerPts.row((p + 1) % 4));
				Core.bitwise_xor(refMarkerPts.row(p), refMarkerPts.row((p + 1) % 4), refMarkerPts.row(p));
			}
		}

		// get data
		ret.code1 += (markerData.get(2, 2)[0] == 0 ? 0 : 1) << 0;
		ret.code1 += (markerData.get(2, 4)[0] == 0 ? 0 : 1) << 1;
		ret.code1 += (markerData.get(4, 4)[0] == 0 ? 0 : 1) << 2;
		ret.code1 += (markerData.get(4, 2)[0] == 0 ? 0 : 1) << 3;

		ret.code2 += (markerData.get(2, 3)[0] == 0 ? 0 : 1) << 0;
		ret.code2 += (markerData.get(3, 4)[0] == 0 ? 0 : 1) << 1;
		ret.code2 += (markerData.get(4, 3)[0] == 0 ? 0 : 1) << 2;
		ret.code2 += (markerData.get(3, 2)[0] == 0 ? 0 : 1) << 3;

		// check parity
		if ((markerData.get(1, 2)[0] != 0 && (ret.code1 >> 1 ^ ret.code2 >> 0) == 0)
				|| (markerData.get(1, 3)[0] != 0 && (ret.code1 >> 0 ^ ret.code1 >> 1) == 0)
				|| (markerData.get(1, 4)[0] != 0 && (ret.code1 >> 0 ^ ret.code2 >> 0) == 0)
				|| (markerData.get(2, 5)[0] != 0 && (ret.code1 >> 2 ^ ret.code2 >> 1) == 0)
				|| (markerData.get(3, 5)[0] != 0 && (ret.code1 >> 1 ^ ret.code1 >> 2) == 0)
				|| (markerData.get(4, 5)[0] != 0 && (ret.code1 >> 1 ^ ret.code2 >> 1) == 0)
				|| (markerData.get(5, 4)[0] != 0 && (ret.code1 >> 3 ^ ret.code2 >> 2) == 0)
				|| (markerData.get(5, 3)[0] != 0 && (ret.code1 >> 2 ^ ret.code1 >> 3) == 0)
				|| (markerData.get(5, 2)[0] != 0 && (ret.code1 >> 2 ^ ret.code2 >> 2) == 0)
				|| (markerData.get(4, 1)[0] != 0 && (ret.code1 >> 0 ^ ret.code2 >> 3) == 0)
				|| (markerData.get(3, 1)[0] != 0 && (ret.code1 >> 3 ^ ret.code1 >> 0) == 0)
				|| (markerData.get(2, 1)[0] != 0 && (ret.code1 >> 3 ^ ret.code2 >> 3) == 0)) {
			return null;
		}

		System.out.println(ret.code1 + ", " + ret.code2);

		// ImUtils.imshow(new JFrame(), markerData, 50);
		ImUtils.imshow(new JFrame(), markerPic, 1);
		return ret;

	}
}

package pers.season.vml.ar;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.util.ImUtils;

public final class MyArUco {
	public final static int WIDTH = 7;
	public final static int HEIGHT = 7;
	public int code1 = 0;
	public int code2 = 0;
	public static double whitePriority = 0.6;
	public MatOfPoint2f pts;

	public MyArUco() {

	}

	public static MyArUco parse(Mat pic, MatOfPoint2f markerPts) {

		int lengthPerBlock = 10;
		int width = (int) WIDTH * lengthPerBlock;
		int height = (int) HEIGHT * lengthPerBlock;
		Mat markerPic = new Mat();
		Mat srcPts = new Mat(4, 1, CvType.CV_32FC2);
		srcPts.put(0, 0, 0, 0, width, 0, width, height, 0, height);
		Mat homo = Imgproc.getPerspectiveTransform(srcPts, markerPts);
		Imgproc.warpPerspective(pic, markerPic, homo, new Size(width, height),
				Imgproc.WARP_INVERSE_MAP | Imgproc.INTER_CUBIC);
		Imgproc.threshold(markerPic, markerPic, 0, 255, Imgproc.THRESH_OTSU);

		Mat markerData = new Mat(HEIGHT, WIDTH, CvType.CV_8U);
		for (int y = 0; y < HEIGHT; y++)
			for (int x = 0; x < WIDTH; x++) {
				int nonZeroPixels = Core.countNonZero(markerPic.submat(y * lengthPerBlock, (y + 1) * lengthPerBlock,
						x * lengthPerBlock, (x + 1) * lengthPerBlock));
				markerData.put(y, x, nonZeroPixels > lengthPerBlock * lengthPerBlock * whitePriority ? 255 : 0);
			}

		if (markerData.width() != WIDTH || markerData.height() != HEIGHT)
			return null;

		MyArUco ret = new MyArUco();
		// check surrounding blocks
		if (Core.countNonZero(markerData.row(0)) != 0 || Core.countNonZero(markerData.row(HEIGHT - 1)) != 0
				|| Core.countNonZero(markerData.col(0)) != 0 || Core.countNonZero(markerData.col(WIDTH - 1)) != 0)
			return null;

		// check corner and adjust rotation
		if ((markerData.get(1, 1)[0] != 0 ? 1 : 0) + (markerData.get(1, WIDTH - 2)[0] != 0 ? 1 : 0)
				+ (markerData.get(HEIGHT - 2, WIDTH - 2)[0] != 0 ? 1 : 0)
				+ (markerData.get(HEIGHT - 2, 1)[0] != 0 ? 1 : 0) != 1)
			return null;
		ret.pts = new MatOfPoint2f(markerPts.clone());
		if (markerData.get(1, WIDTH - 2)[0] != 0) {
			Core.rotate(markerData, markerData, Core.ROTATE_90_COUNTERCLOCKWISE);
			for (int p = 0; p < 3; p++) {
				Core.bitwise_xor(ret.pts.row(p), ret.pts.row((p + 1) % 4), ret.pts.row(p));
				Core.bitwise_xor(ret.pts.row(p), ret.pts.row((p + 1) % 4), ret.pts.row((p + 1) % 4));
				Core.bitwise_xor(ret.pts.row(p), ret.pts.row((p + 1) % 4), ret.pts.row(p));
			}
		} else if (markerData.get(HEIGHT - 2, WIDTH - 2)[0] != 0) {
			Core.rotate(markerData, markerData, Core.ROTATE_180);
			for (int p = 0; p < 2; p++) {
				Core.bitwise_xor(ret.pts.row(p), ret.pts.row((p + 2) % 4), ret.pts.row(p));
				Core.bitwise_xor(ret.pts.row(p), ret.pts.row((p + 2) % 4), ret.pts.row((p + 2) % 4));
				Core.bitwise_xor(ret.pts.row(p), ret.pts.row((p + 2) % 4), ret.pts.row(p));
			}
		} else if (markerData.get(HEIGHT - 2, 1)[0] != 0) {
			Core.rotate(markerData, markerData, Core.ROTATE_90_CLOCKWISE);
			for (int p = 3; p > 0; p--) {
				Core.bitwise_xor(ret.pts.row(p), ret.pts.row((p + 1) % 4), ret.pts.row(p));
				Core.bitwise_xor(ret.pts.row(p), ret.pts.row((p + 1) % 4), ret.pts.row((p + 1) % 4));
				Core.bitwise_xor(ret.pts.row(p), ret.pts.row((p + 1) % 4), ret.pts.row(p));
			}
		}
		//ImUtils.imshow(markerPic);
		// get data
		ret.code1 += (markerData.get(1, 2)[0] == 0 ? 0 : 1) << 7;
		ret.code1 += (markerData.get(1, 3)[0] == 0 ? 0 : 1) << 6;
		ret.code1 += (markerData.get(1, 4)[0] == 0 ? 0 : 1) << 5;
		ret.code1 += (markerData.get(2, 1)[0] == 0 ? 0 : 1) << 4;
		ret.code1 += (markerData.get(2, 2)[0] == 0 ? 0 : 1) << 3;
		ret.code1 += (markerData.get(2, 3)[0] == 0 ? 0 : 1) << 2;
		ret.code1 += (markerData.get(2, 4)[0] == 0 ? 0 : 1) << 1;
		ret.code1 += (markerData.get(2, 5)[0] == 0 ? 0 : 1) << 0;

		ret.code2 += (markerData.get(4, 1)[0] == 0 ? 0 : 1) << 7;
		ret.code2 += (markerData.get(4, 2)[0] == 0 ? 0 : 1) << 6;
		ret.code2 += (markerData.get(4, 3)[0] == 0 ? 0 : 1) << 5;
		ret.code2 += (markerData.get(4, 4)[0] == 0 ? 0 : 1) << 4;
		ret.code2 += (markerData.get(4, 5)[0] == 0 ? 0 : 1) << 3;
		ret.code2 += (markerData.get(5, 2)[0] == 0 ? 0 : 1) << 2;
		ret.code2 += (markerData.get(5, 3)[0] == 0 ? 0 : 1) << 1;
		ret.code2 += (markerData.get(5, 4)[0] == 0 ? 0 : 1) << 0;

		// check parity and checksum
		int parity1 = 0, parity2 = 0;
		for (int b = 0; b < 8; b++) {
			parity1 ^= ret.code1 >> b & 1;
			parity2 ^= ret.code2 >> b & 1;
		}
		int sum = (ret.code1 + ret.code2) & 7;
		int checksum = ((markerData.get(3, 3)[0] == 0 ? 0 : 1) << 2) + ((markerData.get(3, 4)[0] == 0 ? 0 : 1) << 1)
				+ ((markerData.get(3, 5)[0] == 0 ? 0 : 1) << 0);
		if (((markerData.get(3, 1)[0] == 0 ? 0 : 1) != parity1) || ((markerData.get(3, 2)[0] == 0 ? 0 : 1) != parity2)
				|| (checksum != sum)) {
			System.out.println("checksum/parity error with " + ret.code1 + "," + ret.code2);
			// return null;
		}

		return ret;

	}
}

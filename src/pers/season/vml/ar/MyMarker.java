package pers.season.vml.ar;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.util.ImUtils;

public final class MyMarker {
	public final static int WIDTH = 7;
	public final static int HEIGHT = 7;
	public int code1 = 0;
	public int code2 = 0;

	public MyMarker() {

	}

	public static MyMarker load(Mat markerData) {
		
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
		} else if (markerData.get(HEIGHT - 2, WIDTH - 2)[0] != 0) {
			Core.rotate(markerData, markerData, Core.ROTATE_180);
		} else if (markerData.get(HEIGHT - 2, 1)[0] != 0) {
			Core.rotate(markerData, markerData, Core.ROTATE_90_CLOCKWISE);
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
				|| (markerData.get(1, 4)[0] != 0 && (ret.code1 >> 0 ^ ret.code2 >> 0) == 0) ||
				(markerData.get(2, 5)[0] != 0 && (ret.code1 >> 2 ^ ret.code2 >> 1) == 0)
				|| (markerData.get(3, 5)[0] != 0 && (ret.code1 >> 1 ^ ret.code1 >> 2) == 0)
				|| (markerData.get(4, 5)[0] != 0 && (ret.code1 >> 1 ^ ret.code2 >> 1) == 0) ||
				(markerData.get(5, 4)[0] != 0 && (ret.code1 >> 3 ^ ret.code2 >> 2) == 0)
				|| (markerData.get(5, 3)[0] != 0 && (ret.code1 >> 2 ^ ret.code1 >> 3) == 0)
				|| (markerData.get(5, 2)[0] != 0 && (ret.code1 >> 2 ^ ret.code2 >> 2) == 0) ||
				(markerData.get(4, 1)[0] != 0 && (ret.code1 >> 0 ^ ret.code2 >> 3) == 0)
				|| (markerData.get(3, 1)[0] != 0 && (ret.code1 >> 3 ^ ret.code1 >> 0) == 0)
				|| (markerData.get(2, 1)[0] != 0 && (ret.code1 >> 3 ^ ret.code2 >> 3) == 0)) {
			return null;
		}

		System.out.println(ret.code1 + ", " + ret.code2);

		ImUtils.imshow(new JFrame(), markerData, 50);
		return ret;

	}
}

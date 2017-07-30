package pers.season.vml.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import org.opencv.imgcodecs.Imgcodecs;

public class MuctData {
	static int[][] symmetryPoints = { { 0, 14 }, { 1, 13 }, { 2, 12 }, { 3, 11 }, { 4, 10 }, { 5, 9 }, { 6, 8 },
			{ 15, 21 }, { 16, 22 }, { 17, 23 }, { 18, 24 }, { 19, 25 }, { 20, 26 }, { 27, 32 }, { 28, 33 }, { 29, 34 },
			{ 30, 35 }, { 31, 36 }, { 37, 45 }, { 38, 44 }, { 39, 43 }, { 40, 42 }, { 46, 47 }, { 48, 54 }, { 49, 53 },
			{ 50, 52 }, { 55, 59 }, { 56, 58 }, { 60, 62 }, { 65, 63 }, { 68, 72 }, { 69, 73 }, { 70, 74 },
			{ 71, 75 } };

	private static int[] ignorePoints;
	public final static int[] no_ignore = {};
	public final static int[] default_ignore = { 68, 69, 70, 71, 72, 73, 74, 75 };
	public final static int[] futher_ignore = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 68, 69, 70, 71, 72,
			73, 74, 75 };
	public final static int[] max_ignore = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
			21, 22, 23, 24, 25, 26, 27, 28, 29, 30, /* 31, */ 32, 33, 34, 35, /* 36, */ 37, 38, 39, 40, 41, 42, 43, 44,
			45, 46, 47, /* 48, */ 49, 50, 51, 52, 53, /* 54, */ 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
			/* 67, */68, 69, 70, 71, 72, 73, 74, 75 };
	public final static int[] all_ignore = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
			21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
			48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
			75 };

	private static List<String> fileNameLst;
	private static List<float[]> ptsDataLst;

	private static int sampleSize;

	private static String jpgPath;

	private static int ptsCounts;

	public static void init(String jpgPath, String ptsFile, int[] ignore) {
		Arrays.sort(ignore);
		MuctData.ignorePoints = ignore;
		MuctData.ptsCounts = 76 - ignore.length;
		MuctData.jpgPath = jpgPath;
		try {
			fileNameLst = new ArrayList<String>();
			ptsDataLst = new ArrayList<float[]>();
			BufferedReader in = new BufferedReader(new FileReader(ptsFile));
			in.readLine();// skip title
			String line;
			while ((line = in.readLine()) != null) {
				// invalid data
				if (line.startsWith("ir") || line.contains("i434xe-fn"))
					continue;

				// 76 minus 8+15 points, ignoring last 8 points on eyes and 15
				// points bound the face
				float[] ptsData = new float[ptsCounts * 2];
				String[] lineseq = line.split(",");

				boolean brokenData = false;
				for (int i = 0; i < 76 * 2; i++) {
					if (Arrays.binarySearch(ignore, i / 2) < 0) {
						if (lineseq[i + 2].contentEquals("0")) {
							brokenData = true;
							break;
						}
						ptsData[(i / 2 - (-Arrays.binarySearch(ignore, i / 2) - 1)) * 2 + i % 2] = Float
								.parseFloat(lineseq[i + 2]);
					}
				}
				if (!brokenData) {
					fileNameLst.add(lineseq[0]);
					ptsDataLst.add(ptsData);
				}

			}
			projectMirrorShape();

			sampleSize = ptsDataLst.size();
			System.out.println("muct database sample loaded. size = " + sampleSize);
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public static String getFileName(int index) {
		return fileNameLst.get(index);
	}

	public static Mat getJpg(int index) {
		Mat result;
		if (index >= sampleSize / 2) {
			result = Imgcodecs.imread(jpgPath + "/" + fileNameLst.get(index - sampleSize / 2) + ".jpg");
			Core.flip(result, result, 1);
		} else
			result = Imgcodecs.imread(jpgPath + "/" + fileNameLst.get(index) + ".jpg");

		return result;
	}

	public static Mat getGrayJpg(int index) {
		Mat result;
		if (index >= sampleSize / 2) {
			result = Imgcodecs.imread(jpgPath + "/" + fileNameLst.get(index - sampleSize / 2) + ".jpg",
					Imgcodecs.IMREAD_GRAYSCALE);
			Core.flip(result, result, 1);
		} else
			result = Imgcodecs.imread(jpgPath + "/" + fileNameLst.get(index) + ".jpg", Imgcodecs.IMREAD_GRAYSCALE);

		return result;
	}

	public static float[] getPts(int index) {
		return ptsDataLst.get(index);
	}

	public static Mat getPtsMat(int index) {
		float[] data = ptsDataLst.get(index);
		Mat result = new Mat(data.length, 1, CvType.CV_32F);
		result.put(0, 0, data);
		return result;
	}

	public static int getSize() {
		return sampleSize;
	}

	public static int getPtsCounts() {
		return ptsCounts;
	}

	private static void projectMirrorShape() {
		int srcSampleCounts = ptsDataLst.size();
		for (int i = 0; i < srcSampleCounts; i++) {
			float[] src = ptsDataLst.get(i);
			float[] dst = Arrays.copyOf(src, src.length);
			for (int ii = 0; ii < ptsCounts; ii++)
				dst[ii * 2] = 480 - dst[ii * 2];

			for (int ii = 0; ii < symmetryPoints.length; ii++) {
				int[] sp = symmetryPoints[ii];
				// both ignored
				if (Arrays.binarySearch(ignorePoints, sp[0]) >= 0 && (Arrays.binarySearch(ignorePoints, sp[1]) >= 0))
					continue;
				// single one ignored, error
				if ((Arrays.binarySearch(ignorePoints, sp[0]) < 0) ^ (Arrays.binarySearch(ignorePoints, sp[1]) < 0)) {
					System.err.println("shape symmetry & ignoring unpaired.");
					return;
				}

				// both not ignored
				int sp0 = sp[0] - (-Arrays.binarySearch(ignorePoints, sp[0]) - 1);
				int sp1 = sp[1] - (-Arrays.binarySearch(ignorePoints, sp[1]) - 1);

				float temp;
				temp = dst[sp0 * 2];
				dst[sp0 * 2] = dst[sp1 * 2];
				dst[sp1 * 2] = temp;

				temp = dst[sp0 * 2 + 1];
				dst[sp0 * 2 + 1] = dst[sp1 * 2 + 1];
				dst[sp1 * 2 + 1] = temp;

			}
			ptsDataLst.add(dst);

		}

	}

}

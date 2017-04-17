package pers.season.vml.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import org.opencv.imgcodecs.Imgcodecs;

public class MuctData {
	static int[][] symmetryPoints = { { 0, 14 }, { 1, 13 }, { 2, 12 }, { 3, 11 }, { 4, 10 }, { 5, 9 }, { 6, 8 },
			{ 15, 21 }, { 16, 22 }, { 17, 23 }, { 18, 24 }, { 19, 25 }, { 20, 26 }, { 27, 32 }, { 28, 33 }, { 29, 34 },
			{ 30, 35 }, { 31, 36 }, { 37, 45 }, { 38, 44 }, { 39, 43 }, { 40, 42 }, { 46, 47 }, { 48, 54 }, { 49, 53 },
			{ 50, 52 }, { 55, 59 }, { 56, 58 }, { 60, 62 }, { 65, 63 } };
	


	private static List<String> fileNameLst;
	private static List<float[]> ptsDataLst;
	private static int sampleSize;

	private static String jpgPath;
	private static boolean ignoreFaceBoundary;
	private static int ptsCounts;

	public static void init() {
		init("e:/muct/jpg", "e:/muct/muct76-opencv.csv", false);
	}

	public static void init(String jpgPath, String ptsFile, boolean ignoreFaceBoundary) {
		MuctData.ignoreFaceBoundary = ignoreFaceBoundary;
		MuctData.ptsCounts = 76 - 8 - (ignoreFaceBoundary ? 15 : 0);
		MuctData.jpgPath = jpgPath;
		try {
			fileNameLst = new LinkedList<String>();
			ptsDataLst = new LinkedList<float[]>();
			BufferedReader in = new BufferedReader(new FileReader(ptsFile));
			in.readLine();// skip title
			String line;
			while ((line = in.readLine()) != null) {
				// invalid data
				if (line.contains(",0,0,") || line.startsWith("ir") || line.contains("i434xe-fn"))
					continue;

				// 76 minus 8+15 points, ignoring last 8 points on eyes and 15
				// points bound the face
				float[] ptsData = new float[ptsCounts * 2];
				String[] lineseq = line.split(",");

				for (int i = 0; i < ptsData.length; i++) {
					ptsData[i] = Float.parseFloat(lineseq[i + 2 + (ignoreFaceBoundary ? 15 * 2 : 0)]);
				}
				fileNameLst.add(lineseq[0]);
				ptsDataLst.add(ptsData);

			}
			projectMirrorShape();
			sampleSize = ptsDataLst.size();
			System.out.println("muct database sample loaded. size = " + sampleSize);
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

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
				if (ignoreFaceBoundary && (sp[0] < 15 || sp[1] < 15)) {
					continue;
				}
				int sp0 = sp[0] - (ignoreFaceBoundary ? 15 : 0);
				int sp1 = sp[1] - (ignoreFaceBoundary ? 15 : 0);

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

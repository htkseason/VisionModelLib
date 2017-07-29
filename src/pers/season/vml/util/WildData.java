package pers.season.vml.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class WildData {

	protected final static int PTS_FILE_VERSION = 1;
	protected final static int PTS_FILE_NPOINTS = 68;
	protected static List<float[]> ptsDataList = new ArrayList<float[]>();
	protected static List<String> imageFileList = new ArrayList<String>();
	protected static int sampleSize;

	public static void main(String[] args) {
		init("E:\\300w");
		System.loadLibrary("lib/opencv_java320_x64");
		JFrame win = new JFrame();
		for (int i = 0; i < getSize(); i++) {
			Mat pic = getJpg(i);
			Mat pts = getPtsMat(i);
			for (int p = 0; p < pts.rows() / 2; p++)
				Imgproc.circle(pic, new Point(pts.get(p * 2, 0)[0], pts.get(p * 2 + 1, 0)[0]), 5,
						new Scalar(0, 255, 0),5);
			ImUtils.imshow(win, pic, 0.5f);
			try {
				Thread.sleep(300);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public static void init(String path) {
		File pathFile = new File(path);
		if (!pathFile.exists())
			return;
		loadPath(path);
		File[] subFiles = pathFile.listFiles();
		for (File file : subFiles) {
			if (file.isDirectory()) {
				init(file.getAbsolutePath());
			}
		}
		if (ptsDataList.size() != imageFileList.size()) {
			ptsDataList.clear();
			imageFileList.clear();
			System.err.println("pts/image size not match");
		} else
			sampleSize = ptsDataList.size();
	}

	public static Mat getPtsMat(int index) {
		float[] data = ptsDataList.get(index);
		Mat result = new Mat(data.length, 1, CvType.CV_32F);
		result.put(0, 0, data);
		return result;
	}

	public static Mat getGrayJpg(int index) {
		Mat result = Imgcodecs.imread(imageFileList.get(index), Imgcodecs.IMREAD_GRAYSCALE);
		return result;
	}

	public static Mat getJpg(int index) {
		Mat result = Imgcodecs.imread(imageFileList.get(index), Imgcodecs.IMREAD_UNCHANGED);
		return result;
	}

	public static int getSize() {
		return sampleSize;
	}

	protected static void loadPath(String dataPath) {
		File path = new File(dataPath);
		if (!path.exists())
			return;
		File[] subFiles = path.listFiles();
		for (File file : subFiles) {
			if (!file.getName().endsWith(".pts"))
				continue;
			String filePrefix = file.getAbsolutePath().substring(0, file.getAbsolutePath().lastIndexOf('.'));
			String imageFileName = null;
			if (new File(filePrefix + ".jpg").exists())
				imageFileName = filePrefix + ".jpg";
			else if (new File(filePrefix + ".png").exists())
				imageFileName = filePrefix + ".png";
			if (imageFileName == null)
				System.err.println(file.getAbsolutePath() + " image file not exist");
			float[] ptsData = readPtsFile(file, PTS_FILE_VERSION, PTS_FILE_NPOINTS);
			if (ptsData != null) {
				ptsDataList.add(ptsData);
				imageFileList.add(imageFileName);
			}
		}
	}

	protected static float[] readPtsFile(File ptsFile, int expectedVersion, int expectedNPts) {
		if (!ptsFile.exists()) {
			System.err.println("pts file not exist " + ptsFile.toString());
			return null;
		}
		BufferedReader in = null;
		try {
			in = new BufferedReader(new FileReader(ptsFile));

			// check version
			String versionLine = in.readLine().replaceAll(" ", "");
			String[] version = versionLine.split(":");
			if (version.length != 2 || !version[0].toLowerCase().contentEquals("version")
					|| !version[1].contentEquals(new Integer(expectedVersion).toString())) {
				System.err.println("invalid pts file version " + ptsFile.toString());
				return null;
			}

			// check npoints
			String npointsLine = in.readLine().replaceAll(" ", "");
			String[] npoints = npointsLine.split(":");
			if (npoints.length != 2 || !npoints[0].toLowerCase().contentEquals("n_points")
					|| !npoints[1].contentEquals(new Integer(expectedNPts).toString())) {
				System.err.println("invalid pts file npoints " + ptsFile.toString());
				return null;
			}

			// read data
			float[] result = new float[expectedNPts * 2];
			String line = null;
			int pos = 0;
			while ((line = in.readLine()) != null) {
				String[] vals = line.split(" ");
				// check is space>1
				if (vals.length > 2) {
					String[] nvals = new String[2];
					int p = 0;
					for (String str : vals) {
						if (!str.isEmpty()) {
							nvals[p++] = str;
						}
					}
					if (p == 2)
						vals = nvals;
				}
				if (vals.length != 2)
					continue;
				try {
					float x = Float.parseFloat(vals[0]);
					float y = Float.parseFloat(vals[1]);
					result[pos * 2] = x;
					result[pos * 2 + 1] = y;
					pos++;
				} catch (Exception e) {
					continue;
				}
			}
			if (pos == 68)
				return result;
			else {
				System.err.println("invalid pts file data " + ptsFile.toString());
				return null;
			}

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				in.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		System.err.println("invalid pts file " + ptsFile.toString());
		return null;
	}

}

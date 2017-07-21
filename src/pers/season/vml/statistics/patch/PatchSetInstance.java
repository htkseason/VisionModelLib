package pers.season.vml.statistics.patch;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import pers.season.vml.util.ImUtils;

public class PatchSetInstance {

	protected Mat curPts;
	protected PatchSet ps;

	public PatchSetInstance(PatchSet ps) {
		this.ps = ps;
		this.curPts = Mat.zeros(ps.PTS_COUNT, 1, CvType.CV_32F);
	}

	public Mat track(Mat pic, Size searchSize) {
		Mat dstPts = ps.track(pic, curPts, searchSize);
		curPts = dstPts.clone();
		return dstPts;
	}

	public void setCurPts(Mat pts) {
		this.curPts = pts.clone();
	}

	public Mat getCurPts() {
		return curPts.clone();
	}

}

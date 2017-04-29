package pers.season.vml.statistics.regressor;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import pers.season.vml.util.ImUtils;

public class RegressorSetInstance {

	protected Mat curPts;
	protected RegressorSet rs;

	public RegressorSetInstance(RegressorSet rs) {
		this.rs = rs;
		this.curPts = Mat.zeros(rs.PTS_COUNT, 1, CvType.CV_32F);
	}

	public Mat track(Mat pic, Size searchSize) {
		Mat dstPts = rs.track(pic, curPts, searchSize);
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

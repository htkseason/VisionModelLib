package pers.season.vml.statistics.regressor;

import org.opencv.core.Mat;
import org.opencv.core.Size;

public class RegressorSetInstance extends RegressorSet {
	protected Mat patches;
	protected Mat curPts;
	protected Size patchSize;
	protected Mat refShape;

	protected RegressorSetInstance() {

	}

	public static RegressorSetInstance load(Mat patches, Size patchSize, Mat refShape) {
		RegressorSetInstance rsi = new RegressorSetInstance();
		rsi.patches = patches.clone();
		rsi.patchSize = patchSize.clone();
		rsi.refShape = refShape.clone();
		return rsi;
	}

	public Mat track(Mat pic, Size searchSize) {
		Mat dstPts = track(patches, pic, curPts, refShape, patchSize, searchSize);
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

package pers.season.vml.statistics.regressor;

import org.opencv.core.Mat;
import org.opencv.core.Size;

public class RegressorSetInstance extends RegressorSet {
	protected Mat patches;
	protected Mat curPts;
	protected Size patchSize;
	protected Mat refShape;

	public void load(Mat patches, Size patchSize, Mat refShape) {
		this.patches = patches.clone();
		this.patchSize = patchSize.clone();
		this.refShape = refShape.clone();
	}
	
	public void setInitPts(Mat pts) {
		this.curPts = pts.clone();
	}

	public Mat track(Mat pic, Size searchSize) {
		Mat dstPts = track(patches, pic, curPts, refShape, patchSize, searchSize);
		curPts = dstPts.clone();
		return dstPts;
	}

}

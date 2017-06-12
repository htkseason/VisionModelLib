package pers.season.vml.ar;

import org.opencv.core.Core;
import org.opencv.core.Mat;

public class MotionFilter {

	protected double momentum;
	protected Mat motion;
	protected int paramc;

	public MotionFilter(int paramc, double momentum) {
		this.paramc = paramc;
		this.momentum = momentum;
	}

	public Mat next(Mat data) {
		if (data.total() * data.channels() != paramc)
			return null;
		if (motion == null) {
			motion = data.clone();
			return motion;
		} else {
			Core.addWeighted(motion, momentum, data, 1 - momentum, 0, motion);
			return motion;
		}
	}

	public void reset() {
		motion = null;
	}
}

package pers.season.vml.util;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.video.KalmanFilter;

public class Kalman {
	protected KalmanFilter kalman;
	protected Mat measurement;
	protected int paramc;
	protected Mat offsets;

	public Kalman(int paramc) {
		this.paramc = paramc;
		init();
	}

	protected void init() {
		kalman = new KalmanFilter(paramc * 2, paramc);
		measurement = Mat.zeros(paramc, 1, CvType.CV_32F);
		Core.setIdentity(kalman.get_measurementMatrix());
		Core.setIdentity(kalman.get_processNoiseCov(), new Scalar(1e-5));
		Core.setIdentity(kalman.get_measurementNoiseCov(), new Scalar(1e-1));
		Core.setIdentity(kalman.get_errorCovPost(), new Scalar(1));
		kalman.set_transitionMatrix(Mat.zeros(new Size(paramc * 2, paramc * 2), CvType.CV_32F));

		// init transitionMatrix
		for (int i = 0; i < kalman.get_transitionMatrix().rows(); i++) {
			kalman.get_transitionMatrix().put(i, i, 1);
			if (i + kalman.get_transitionMatrix().cols() / 2 < kalman.get_transitionMatrix().cols())
				kalman.get_transitionMatrix().put(i, i + kalman.get_transitionMatrix().cols() / 2, 1);
		}
	}

	public void reset() {
		init();
		offsets = null;
	}

	public void correct(Mat params) {
		if (params.total() * params.channels() != paramc)
			return;
		params.copyTo(measurement);
		if (offsets == null)
			offsets = measurement.clone();
		Core.subtract(measurement, offsets, measurement);
		
		kalman.correct(measurement);

	}

	public Mat predict() {
		Mat result = kalman.predict().clone();
		ImUtils.printMat(result);
		Core.add(result.rowRange(0, paramc), offsets, result.rowRange(0, paramc));
		ImUtils.printMat(result);
		return result;
	}

	public int getParamCount() {
		return paramc;
	}
}

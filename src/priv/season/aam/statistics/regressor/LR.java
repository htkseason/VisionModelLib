package priv.season.aam.statistics.regressor;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class LR {

	public Mat theta;
	public Mat trainSample, response;
	public double lambda;

	public LR() {

	}

	public void train(Mat sample, Mat response) {
		this.theta = Mat.zeros(sample.cols() + 1, 1, CvType.CV_64F);
		this.trainSample = Mat.ones(sample.rows(), sample.cols() + 1, CvType.CV_64F);
		sample.copyTo(trainSample.colRange(1, trainSample.cols()));
		this.response = response.clone();
	}



	public double getCost() {
		Mat hx = getHx();
		Mat t1 = new Mat(), t2 = new Mat(), t3 = new Mat();
		double result;

		int m = trainSample.rows();



		// positive sample cost.
		// y*log(hz) --> t1
		Core.log(hx, t1);
		Core.multiply(response, t1, t1);

		// negative sample cost.
		// (1-y)*log(1-hz) --> t2
		Core.subtract(Mat.ones(response.size(), response.type()), response, t3);
		Core.subtract(Mat.ones(hx.size(), hx.type()), hx, t2);
		Core.log(t2, t2);
		Core.multiply(t3, t2, t2);

		// combine
		Core.add(t1, t2, hx);
		result = Core.sumElems(hx).val[0];
		result /= -m;

		// regularization
		Core.multiply(theta, theta, t1);
		double t = Core.sumElems(t1.rowRange(1, t1.rows())).val[0];
		t = (lambda / (2 * m)) * t;
		result = result + t;


		return result;

	}
	
	protected Mat getHx() {
		Mat hx = new Mat();
		Core.gemm(trainSample, theta, 1, new Mat(), 0, hx);
		Core.multiply(hx, new Scalar(-1), hx);
		Core.exp(hx, hx);
		Core.add(hx, new Scalar(1), hx);
		Core.divide(1, hx, hx);
		return hx;
	}

	protected void updataTheta(int sampleSize) {

	}

}

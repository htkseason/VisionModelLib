package pers.season.vml.ml;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import pers.season.vml.util.CwMat;
import pers.season.vml.util.ImUtils;

public class LinearRegression {

	public Mat theta;
	public Mat X, y;
	public double lambda;
	protected final int CORE_COUNTS = Runtime.getRuntime().availableProcessors();
	protected ExecutorService threadPool = Executors.newCachedThreadPool();

	public LinearRegression() {

	}

	public void setData(Mat sample, Mat response, double lambda, boolean normalize) {
		this.theta = Mat.zeros(sample.cols() + 1, 1, CvType.CV_32F);
		this.X = Mat.ones(sample.rows(), sample.cols() + 1, CvType.CV_32F);
		sample.copyTo(X.colRange(1, X.cols()));
		if (response.type() != CvType.CV_32F)
			response.convertTo(response, CvType.CV_32F);
		this.y = response.clone();
		this.lambda = lambda;
		if (normalize)
			normalize();
	}

	private void normalize() {
		Mat Xt = X.colRange(1, X.cols());
		Mat mean = Mat.zeros(1, Xt.cols(), CvType.CV_32F);
		for (int i = 0; i < Xt.rows(); i++)
			Core.add(mean, Xt.row(i), mean);
		Core.divide(mean, new Scalar(Xt.rows()), mean);
		for (int i = 0; i < Xt.rows(); i++) {
			Core.subtract(Xt.row(i), mean, Xt.row(i));
			Core.normalize(Xt.row(i), Xt.row(i));
		}
	}

	public double getSampleCost() {
		Mat cost = new Mat();
		Core.gemm(X, theta, 1, new Mat(), 0, cost);
		Core.subtract(cost, y, cost);
		Core.gemm(cost.t(), cost, 1, new Mat(), 0, cost);
		return Math.sqrt(cost.get(0, 0)[0] / X.rows());
	}

	public double getCost() {
		Mat reg = new Mat();
		Core.gemm(theta.t(), theta, 1, new Mat(), 0, reg);
		double reg_v = reg.get(0, 0)[0] - theta.get(0, 0)[0] * theta.get(0, 0)[0];

		Mat cost = new Mat();
		Core.gemm(X, theta, 1, new Mat(), 0, cost);
		Core.subtract(cost, y, cost);
		Core.gemm(cost.t(), cost, 1, new Mat(), 0, cost);
		double cost_v = cost.get(0, 0)[0];

		return (cost_v + lambda * reg_v) / (2 * X.rows());
	}

	public Mat getGradient(int batchSize) {
		int rndSampleCount = (int) (Math.random() * (X.rows() / batchSize));
		Mat tX = X.rowRange(rndSampleCount * batchSize, (rndSampleCount + 1) * batchSize);
		Mat tY = y.rowRange(rndSampleCount * batchSize, (rndSampleCount + 1) * batchSize);
		Mat comt = new Mat();
		Core.gemm(tX, theta, 1, new Mat(), 0, comt);
		Core.subtract(comt, tY, comt);
		Mat g = new Mat(theta.rows(), 1, CvType.CV_32F);

		Semaphore sema = new Semaphore(0);

		int assignmentCounts = theta.rows() / CORE_COUNTS;
		for (int threadIndex = 0; threadIndex < CORE_COUNTS; threadIndex++) {
			final int curThreadIndex = threadIndex;
			threadPool.execute(new Runnable() {
				@Override
				public void run() {
					int bottomBorder = (curThreadIndex + 1) * assignmentCounts;
					if (curThreadIndex == CORE_COUNTS - 1)
						bottomBorder = theta.rows();
					Mat t = new Mat();
					for (int i = curThreadIndex * assignmentCounts; i < bottomBorder; i++) {
						Core.multiply(comt, tX.col(i), t);
						double sum = Core.sumElems(t).val[0];
						if (i != 0)
							sum += lambda * theta.get(i, 0)[0];
						sum /= tX.rows();
						g.put(i, 0, sum);
					}
					sema.release();
				}
			});
		}
		try {
			sema.acquire(CORE_COUNTS);
		} catch (InterruptedException e) {

		}

		return g;
	}

}

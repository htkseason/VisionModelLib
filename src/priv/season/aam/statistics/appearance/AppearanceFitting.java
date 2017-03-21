package priv.season.aam.statistics.appearance;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import priv.season.aam.statistics.shape.ShapeModel;
import priv.season.aam.statistics.texture.TextureModel;

public class AppearanceFitting extends AppearanceInstance {

	public Mat pic;

	public AppearanceFitting(Mat pic, Mat shapeZ, Mat textureZ) {
		super(shapeZ, textureZ);
		this.pic = pic;
	}

	public Mat getGradient() {

		Mat cost_U = new Mat();
		Mat cost = getCost();
		for (int i = 0; i < Z_SIZE; i++) {
			double gap = 0;

			if (i < 2)
				gap = (Math.random() > 0.5 ? 1 : -1) * 50;
			else if (i < 4)
				gap = (Math.random() > 0.5 ? 1 : -1) * 50;
			else
				gap = (Math.random() > 0.5 ? 1 : -1) * e.get(i - 4, 0)[0] * 0.50;

			// gaus kernal?
			// double sigma =1;
			// gap =
			// 100*(1/(sigma*Math.sqrt(2*Math.PI)))*Math.exp(-(gap*gap)/(2*sigma));

			Z.put(i, 0, Z.get(i, 0)[0] + gap);
			Mat temp_cost = getCost();
			Core.subtract(temp_cost, cost, temp_cost);
			Core.multiply(temp_cost, new Scalar(1 / gap), temp_cost);
			Z.put(i, 0, Z.get(i, 0)[0] - gap);
			cost_U.push_back(temp_cost.t());

		}

		cost_U = cost_U.t();
		Mat result = new Mat();
		Core.gemm(cost_U.t(), cost_U, 1, new Mat(), 0, result);
		Core.invert(result, result);
		Core.gemm(result, cost_U.t(), 1, new Mat(), 0, result);

		Core.gemm(result, cost, 1, new Mat(), 0, result);

		return result;
	}

	public void updata(Mat gradient) {
		int s = 0;
		Core.subtract(Z.rowRange(s, s + gradient.rows()), gradient, Z.rowRange(s, s + gradient.rows()));
	}

	public Mat getCost() {
		Mat cost = new Mat();
		Mat X = getXfromZ(Z);
		double scale = ShapeModel.getScale(X);

		Core.multiply(X.rowRange(4, ShapeModel.Z_SIZE), new Scalar(scale), X.rowRange(4, ShapeModel.Z_SIZE));

		Core.subtract(TextureModel.getXfromZ(X.rowRange(ShapeModel.Z_SIZE, X.rows())),
				TextureModel.getNormFace(pic, ShapeModel.getXfromZ(X.rowRange(0, ShapeModel.Z_SIZE))).reshape(1,
						TextureModel.X_SIZE),
				cost);

		return cost;
	}
}

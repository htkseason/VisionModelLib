package pers.season.vml.statistics.sdm;

import java.io.File;
import java.util.Random;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.RTrees;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import pers.season.vml.ml.LearningParams;
import pers.season.vml.ml.LinearRegression;
import pers.season.vml.statistics.patch.PatchSet;
import pers.season.vml.statistics.shape.ShapeInstance;
import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public class SdmTrain {

	public static void train(Mat refShape, Size patchSize) {
		ShapeModel sm = ShapeModel.load("models/shape/", "V", "Z_e");
		ShapeInstance shape = new ShapeInstance(sm);
		JFrame win = new JFrame();

		Mat inputData = Mat.zeros(SdmHogDescriptor.DESCRIPTION_SIZE * refShape.rows() / 2, 100, CvType.CV_32F);
		Mat outputData = Mat.zeros(refShape.rows(), 100, CvType.CV_32F);
		MatOfFloat mof = new MatOfFloat();
		for (int i = 0; i < 100; i++) {

			Mat srcPic = MuctData.getGrayJpg(i);
			Mat srcPts = MuctData.getPtsMat(i);

			// project non-rigid transformation
			shape.setFromPts(srcPts);
			shape.Z.rowRange(4, shape.Z.rows()).setTo(new Scalar(0));
			// random orientation
			//shape.setRadian((Math.random() * 2 - 1) * 20 / 180 * Math.PI);
			// random scale
			double scale = shape.getScale();
			scale = scale * (Math.random() * 0.5 + 0.75);
			//shape.setScale(scale);
			// random location
			Point loc = shape.getOffset();
			double offsetAngle = Math.random() * 2 * Math.PI;
			double offsetDistance = Math.random() * 0.2 * (scale / sm.getScalePerPixel());
			//shape.setOffset(new Point(loc.x + Math.cos(offsetAngle) * offsetDistance,
					//loc.y + Math.sin(offsetAngle) * offsetDistance));

			Mat srcPtsFalse = shape.getX();
			Mat R = PatchSet.getPtsAffineTrans(srcPtsFalse, refShape, srcPic.width() / 2, srcPic.height() / 2);
			Mat affPtsFalse = PatchSet.warpPtsAffine(srcPtsFalse, R);
			Mat affPtsTrue = PatchSet.warpPtsAffine(srcPts, R);
			Mat affPic = new Mat();
			Imgproc.warpAffine(srcPic, affPic, R, srcPic.size());
			
			Core.subtract(affPtsTrue, affPtsFalse, outputData.colRange(i * 2, i * 2 + 1));
			System.out.println(Core.norm( outputData.colRange(i * 2, i * 2 + 1)));
			for (int p = 0; p < affPtsFalse.rows() / 2; p++) {
				SdmHogDescriptor.compute(affPic,
						new Point(affPtsFalse.get(p * 2, 0)[0], affPtsFalse.get(p * 2 + 1, 0)[0]), patchSize, mof);
				mof.copyTo(inputData.submat(p * SdmHogDescriptor.DESCRIPTION_SIZE,
						(p + 1) * SdmHogDescriptor.DESCRIPTION_SIZE, i * 2, i * 2 + 1));

				SdmHogDescriptor.compute(affPic,
						new Point(affPtsTrue.get(p * 2, 0)[0], affPtsTrue.get(p * 2 + 1, 0)[0]), patchSize, mof);
				mof.copyTo(inputData.submat(p * SdmHogDescriptor.DESCRIPTION_SIZE,
						(p + 1) * SdmHogDescriptor.DESCRIPTION_SIZE, i * 2 + 1, i * 2 + 2));
				
				// Imgproc.circle(affPic, new Point(affPtsFalse.get(p * 2, 0)[0],
				// affPtsFalse.get(p * 2 + 1, 0)[0]), 2,
				// new Scalar(255));

			}

		}

		String savePath = "models/sdm_all_0sample_pca/";

		System.out.println(inputData.size());
		System.out.println(outputData.size());

		Mat pcaMean = new Mat();
		Mat pcaVec = new Mat();
		Core.PCACompute(inputData.t(), pcaMean, pcaVec, 0.98);
		Core.PCAProject(inputData.t(), pcaMean, pcaVec, inputData);
				ImUtils.saveMat(pcaMean, savePath + "pcaMean");
		ImUtils.saveMat(pcaVec, savePath + "pcaVec");

		inputData = inputData.t();
		System.out.println(inputData.size());
		System.out.println(outputData.size());

		for (int i = 0; i < outputData.rows(); i++) {
			System.out.println("training regressor " + i + "/" + outputData.rows() + " ...");
			LinearRegression lr = new LinearRegression();
			lr.setData(inputData.t(), outputData.row(i).t(), 1, false);
			if (new File(savePath + "theta_" + i).exists())
				lr.theta = ImUtils.loadMat(savePath + "theta_" + i);
			double delta = 0.001;
			for (int iter = 0; iter <= 10000; iter++) {
				if (iter % 1000 == 0) {
					double cost = lr.getSampleCost();
					System.out.println("iter = " + iter + "\t\tcost = " + cost);
				}

				Mat g_ori = lr.getGradient(500);

				Mat g = new Mat();
				Core.multiply(g_ori, new Scalar(delta), g);
				Core.subtract(lr.theta, g, lr.theta);

				System.gc();

			}
			ImUtils.saveMat(lr.theta, savePath + "theta_" + i);
		}
	}

	private void trainWithSVR(Mat inputData, Mat outputData) {

		for (int i = 0; i < inputData.cols(); i++) {

		}
		// 定义训练集点a{10.0, 10.0} 和 点b{-10.0, -10.0}，对应lable为{1.0, -1.0}
		svm_node pa0 = new svm_node();
		pa0.index = 0;
		pa0.value = 10.0;
		svm_node pa1 = new svm_node();
		pa1.index = -1;
		pa1.value = 10.0;
		svm_node pb0 = new svm_node();
		pb0.index = 0;
		pb0.value = -10.0;
		svm_node pb1 = new svm_node();
		pb1.index = -1;
		pb1.value = -10.0;
		svm_node[] pa = { pa0, pa1 }; // 点a
		svm_node[] pb = { pb0, pb1 }; // 点b
		svm_node[][] datas = { pa, pb }; // 训练集的向量表
		double[] lables = { 23.0, -23.0 }; // a,b 对应的lable

		// 定义svm_problem对象
		svm_problem problem = new svm_problem();
		problem.l = 2; // 向量个数
		problem.x = datas; // 训练集向量表
		problem.y = lables; // 对应的lable数组

		// 定义svm_parameter对象
		svm_parameter param = new svm_parameter();
		param.svm_type = svm_parameter.NU_SVR;
		param.kernel_type = svm_parameter.LINEAR;
		param.cache_size = 100;
		param.nu = 0.5;
		param.eps = 0.00001;
		param.C = 1;

		// 训练SVM分类模型
		System.out.println(svm.svm_check_parameter(problem, param)); // 如果参数没有问题，则svm.svm_check_parameter()函数返回null,否则返回error描述。
		svm_model model = svm.svm_train(problem, param); // svm.svm_train()训练出SVM分类模型

		// 定义测试数据点c
		svm_node pc0 = new svm_node();
		pc0.index = 0;
		pc0.value = 1;
		svm_node pc1 = new svm_node();
		pc1.index = -1;
		pc1.value = 1;
		svm_node[] pc = { pc0, pc1 };

		// 预测测试数据的lable
		System.out.println(svm.svm_predict(model, pc));
	}

}

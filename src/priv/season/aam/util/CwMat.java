package priv.season.aam.util;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class CwMat extends Mat implements Externalizable {

	public CwMat(Mat mat) {
		super(mat.nativeObj);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		int type = in.readInt();
		int rows = in.readInt();
		int cols = in.readInt();
		double[] data = new double[rows * cols * CvType.channels(type)];
		Mat result = new Mat(rows, cols, type);
		result.put(0, 0, data);
		this.setTo(result);
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		Object data = null;
		switch(depth()) {
		case CvType.CV_32F:
			data = new float[rows() * cols() * channels()];
			get(0, 0, (float[]) data);
			break;
		case CvType.CV_64F:
			data = new double[rows() * cols() * channels()];
			get(0, 0, (double[]) data);
			break;
		case CvType.CV_8U:
			data = new byte[rows() * cols() * channels()];
			get(0, 0, (byte[]) data);
			break;
		}
		
		out.writeObject(type());
		out.writeObject(rows());
		out.writeObject(cols());
		out.writeObject(data);
	}

}

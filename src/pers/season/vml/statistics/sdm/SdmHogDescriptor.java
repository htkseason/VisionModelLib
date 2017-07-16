package pers.season.vml.statistics.sdm;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.HOGDescriptor;

public class SdmHogDescriptor {

	public final static int DESCRIPTION_SIZE = 128; 
	public static void compute(Mat pic, Point pt, Size size, Mat des) {

		Rect roi = new Rect((int) pt.x - (int) size.width / 2, (int) pt.y - (int) size.height / 2, (int) size.width,
				(int) size.height);
		if (roi.x < 0) {
			roi.width += roi.x;
			roi.x = 0;
		}
		if (roi.y < 0) {
			roi.height += roi.y;
			roi.y = 0;
		}
		if (roi.x + roi.width > pic.width())
			roi.width = pic.width() - roi.x;

		if (roi.y + roi.height > pic.height())
			roi.height = pic.height() - roi.y;

		HOGDescriptor hog = new HOGDescriptor(roi.size(), roi.size(), roi.size(),
				new Size(roi.size().width / 4, roi.size().height / 4), 8);
		hog.compute(pic.submat(roi), (MatOfFloat) des);

	}

}

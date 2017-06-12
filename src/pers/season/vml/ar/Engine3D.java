package pers.season.vml.ar;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import javax.swing.JFrame;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.jogamp.opengl.GL2;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLCapabilities;
import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.GLProfile;
import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.glu.GLU;
import com.jogamp.opengl.util.FPSAnimator;
import com.jogamp.opengl.util.texture.Texture;
import com.jogamp.opengl.util.texture.TextureData;

import pers.season.vml.util.ImUtils;

public class Engine3D {
	Mat rvec;
	Mat tvec;
	Mat pic;
	Mat camMat;
	Mat templatePic;
	Mat templatePic_resized;
	GLCanvas glcanvas;
	int[] textureIndexs;

	float scale = 200;

	public GLEventListener gel = new GLEventListener() {

		@Override
		public void init(GLAutoDrawable drawable) {
			final GL2 gl = drawable.getGL().getGL2();
			int texCount = 2;
			textureIndexs = new int[texCount];
			gl.glGenTextures(texCount, textureIndexs, 0);
			for (int i = 0; i < textureIndexs.length; i++) {
				gl.glBindTexture(GL2.GL_TEXTURE_2D, textureIndexs[i]);
				gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MIN_FILTER, GL2.GL_LINEAR);
				gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MAG_FILTER, GL2.GL_LINEAR);
			}
		}

		@Override
		public void display(GLAutoDrawable drawable) {
			final GL2 gl = drawable.getGL().getGL2();

			gl.glClear(GL2.GL_DEPTH_BUFFER_BIT | GL2.GL_COLOR_BUFFER_BIT);
			if (pic == null)
				return;

			drawBackground(gl);
			// =====

			if (rvec == null || tvec == null)
				return;

			gl.glMatrixMode(GL2.GL_PROJECTION);
			gl.glLoadMatrixf(ImUtils.get32FMatData(buildProjectionMatrix(camMat, pic.width(), pic.height())), 0);

			gl.glMatrixMode(GL2.GL_MODELVIEW);
			gl.glLoadIdentity();

			Mat rmat = new Mat();
			Mat mv = Mat.zeros(4, 4, CvType.CV_32F);
			Calib3d.Rodrigues(rvec, rmat);
			rmat.t().copyTo(mv.submat(0, 3, 0, 3));
			mv.put(3, 0, -tvec.get(0, 0)[0] / scale, -tvec.get(1, 0)[0] / scale, -tvec.get(2, 0)[0] / scale, 1);
			gl.glLoadMatrixf(ImUtils.get32FMatData(mv), 0);
			drawCoord(gl);
			drawTemplatePlane(gl);
			// =====
			gl.glFlush();
		}

		@Override
		public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {

		}

		@Override
		public void dispose(GLAutoDrawable drawable) {

		}

	};

	public void update(Mat pic, Mat rvec, Mat tvec) {
		if (pic != null)
			this.pic = pic.clone();
		if (rvec != null)
			this.rvec = rvec.clone();
		else
			this.rvec = null;
		if (tvec != null)
			this.tvec = tvec.clone();
		else
			this.tvec = null;

		glcanvas.repaint();
	}

	public Engine3D(Mat camMat, Mat templatePic) {
		this.camMat = camMat.clone();
		this.templatePic = templatePic.clone();
		this.templatePic_resized = new Mat();
		Imgproc.resize(templatePic, this.templatePic_resized, new Size(256, 256));
		final GLProfile profile = GLProfile.get(GLProfile.GL2);
		GLCapabilities capabilities = new GLCapabilities(profile);

		glcanvas = new GLCanvas(capabilities);
		glcanvas.addGLEventListener(gel);
		glcanvas.setSize(800, 600);

		final JFrame frame = new JFrame();

		frame.getContentPane().add(glcanvas);
		frame.setSize(frame.getContentPane().getPreferredSize());
		frame.setVisible(true);

	}

	protected static Mat buildProjectionMatrix(Mat camMat, int screen_width, int screen_height) {
		Mat result = Mat.zeros(4, 4, CvType.CV_32F);
		float nearPlane = 0.01f; // Near clipping distance
		float farPlane = 100.0f; // Far clipping distance

		// Camera parameters
		float f_x = (float) camMat.get(0, 0)[0];
		float f_y = (float) camMat.get(1, 1)[0];
		float c_x = (float) camMat.get(0, 2)[0];
		float c_y = (float) camMat.get(1, 2)[0];

		result.put(0, 0, -2.0f * f_x / screen_width);
		result.put(1, 1, 2.0f * f_y / screen_height);
		result.put(2, 0, 2.0f * c_x / screen_width - 1.0f, 2.0f * c_y / screen_height - 1.0f,
				-(farPlane + nearPlane) / (farPlane - nearPlane), -1.0f);
		result.put(3, 2, -2.0f * farPlane * nearPlane / (farPlane - nearPlane));
		return result;
	}

	protected void drawBackground(GL2 gl) {

		byte[] data = ImUtils.get8UMatData(pic);

		gl.glPixelStorei(GL2.GL_PACK_ALIGNMENT, 1);
		gl.glBindTexture(GL2.GL_TEXTURE_2D, textureIndexs[0]);
		gl.glTexImage2D(GL2.GL_TEXTURE_2D, 0, GL2.GL_RGB, pic.width(), pic.height(), 0, GL2.GL_BGR,
				GL2.GL_UNSIGNED_BYTE, ByteBuffer.wrap(data));

		float[] bgTextureVertices = { 0, 0, pic.width(), 0, 0, pic.height(), pic.width(), pic.height() };
		float[] bgTextureCoords = { 1, 0, 1, 1, 0, 0, 0, 1 };
		float[] proj = { 0, -2.f / pic.width(), 0, 0, -2.f / pic.height(), 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1 };
		gl.glMatrixMode(GL2.GL_PROJECTION);
		gl.glLoadMatrixf(proj, 0);
		gl.glMatrixMode(GL2.GL_MODELVIEW);
		gl.glLoadIdentity();
		gl.glEnable(GL2.GL_TEXTURE_2D);
		gl.glBindTexture(GL2.GL_TEXTURE_2D, textureIndexs[0]);
		gl.glEnableClientState(GL2.GL_VERTEX_ARRAY);
		gl.glEnableClientState(GL2.GL_TEXTURE_COORD_ARRAY);

		gl.glVertexPointer(2, GL2.GL_FLOAT, 0, makeFloatBuffer(bgTextureVertices));
		gl.glTexCoordPointer(2, GL2.GL_FLOAT, 0, makeFloatBuffer(bgTextureCoords));

		gl.glColor4f(1, 1, 1, 1);
		gl.glDrawArrays(GL2.GL_TRIANGLE_STRIP, 0, 4);

		gl.glDisableClientState(GL2.GL_VERTEX_ARRAY);
		gl.glDisableClientState(GL2.GL_TEXTURE_COORD_ARRAY);
		gl.glDisable(GL2.GL_TEXTURE_2D);
	}

	protected float tempalteRot = 0.0f;

	protected void drawTemplatePlane(GL2 gl) {

		byte[] data = ImUtils.get8UMatData(templatePic_resized);

		gl.glPixelStorei(GL2.GL_PACK_ALIGNMENT, 1);
		gl.glBindTexture(GL2.GL_TEXTURE_2D, textureIndexs[1]);
		gl.glTexImage2D(GL2.GL_TEXTURE_2D, 0, GL2.GL_RGB, templatePic_resized.width(), templatePic_resized.height(), 0,
				GL2.GL_BGR, GL2.GL_UNSIGNED_BYTE, ByteBuffer.wrap(data));

		float[] bgTextureVertices = { 0, 0, 1.5f, -0.75f, 0, 0.75f, 0.75f, 0, 0.75f, 0, 0, 0 };
		float[] bgTextureCoords = { 0, 0, 1, 0, 0, 1, 1, 1 };
		gl.glMatrixMode(GL2.GL_MODELVIEW);
		gl.glPushMatrix();
		gl.glTranslatef(-templatePic.width() / scale / 2f, -templatePic.height() / scale / 2f, 0);
		gl.glRotatef(tempalteRot = tempalteRot + 10, 0, 0, 1);

		gl.glEnable(GL2.GL_TEXTURE_2D);
		gl.glBindTexture(GL2.GL_TEXTURE_2D, textureIndexs[1]);
		gl.glEnableClientState(GL2.GL_VERTEX_ARRAY);
		gl.glEnableClientState(GL2.GL_TEXTURE_COORD_ARRAY);

		gl.glVertexPointer(3, GL2.GL_FLOAT, 0, makeFloatBuffer(bgTextureVertices));
		gl.glTexCoordPointer(2, GL2.GL_FLOAT, 0, makeFloatBuffer(bgTextureCoords));

		gl.glColor4f(1, 1, 1, 1);
		gl.glDrawArrays(GL2.GL_TRIANGLE_STRIP, 0, 4);

		gl.glDisableClientState(GL2.GL_VERTEX_ARRAY);
		gl.glDisableClientState(GL2.GL_TEXTURE_COORD_ARRAY);
		gl.glDisable(GL2.GL_TEXTURE_2D);

		gl.glPopMatrix();
	}

	protected static void drawCoord(GL2 gl) {
		float lineX[] = { 0, 0, 0, -1, 0, 0 };
		float lineY[] = { 0, 0, 0, 0, -1, 0 };
		float lineZ[] = { 0, 0, 0, 0, 0, 1 };

		gl.glLineWidth(2);

		gl.glBegin(GL2.GL_LINES);

		gl.glColor3f(1.0f, 0.0f, 0.0f);
		gl.glVertex3fv(lineX, 0);
		gl.glVertex3fv(lineX, 3);

		gl.glColor3f(0.0f, 1.0f, 0.0f);
		gl.glVertex3fv(lineY, 0);
		gl.glVertex3fv(lineY, 3);

		gl.glColor3f(0.0f, 0.0f, 1.0f);
		gl.glVertex3fv(lineZ, 0);
		gl.glVertex3fv(lineZ, 3);

		gl.glEnd();
	}

	protected static FloatBuffer makeFloatBuffer(float[] arr) {
		ByteBuffer bb = ByteBuffer.allocateDirect(arr.length * 4);
		bb.order(ByteOrder.nativeOrder());
		FloatBuffer fb = bb.asFloatBuffer();
		fb.put(arr);
		fb.position(0);
		return fb;
	}
}
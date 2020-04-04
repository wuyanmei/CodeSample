package com.white.imagesobelfilter;
//import org.opencv.android.BaseLoaderCallback;
//import org.opencv.android.LoaderCallbackInterface;
//import org.opencv.android.OpenCVLoader;

import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.view.Menu;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import static android.Manifest.permission.CAMERA;
import static android.Manifest.permission.READ_EXTERNAL_STORAGE;

public class ImageSobelFilter extends Activity {
	private Button chooseButton, sobelGPU_CPU, result_button;
	private ImageView primaryImageView, sobel_gpu_image, sobel_cpu_image,
			sobel_gray_image, sobel_differ_image;
	private TextView primary_text, gray_text, cpu_text, gpu_text, differ_text;
	private int RESULT_LOAD_IMAGE = 3;
	private String imagePathString, resultString;
	private nativeSobelFilter t;
//	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
//		@Override
//		public void onManagerConnected(int status) {
//			switch (status) {
//			case LoaderCallbackInterface.SUCCESS: {
//			}
//				break;
//			default: {
//				super.onManagerConnected(status);
//			}
//				break;
//			}
//		}
//	};

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
//		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,
//				mLoaderCallback);
		setContentView(R.layout.activity_image_sobel_filter);
		
		chooseButton = (Button) findViewById(R.id.choose_button);
		sobelGPU_CPU = (Button) findViewById(R.id.sobel_gpu_cpu);
		result_button = (Button) findViewById(R.id.result_button);
		primaryImageView = (ImageView) findViewById(R.id.primary_image);
		sobel_gpu_image = (ImageView) findViewById(R.id.gpu_image);
		sobel_cpu_image = (ImageView) findViewById(R.id.cpu_image);
		sobel_gray_image = (ImageView) findViewById(R.id.gray_image);
		sobel_differ_image = (ImageView) findViewById(R.id.differ_image);

		primary_text = (TextView) findViewById(R.id.primary_image_text);
		gray_text = (TextView) findViewById(R.id.gray_image_text);
		cpu_text = (TextView) findViewById(R.id.cpu_image_text);
		gpu_text = (TextView) findViewById(R.id.gpu_image_text);
		differ_text = (TextView) findViewById(R.id.differ_image_text);

		sobelGPU_CPU.setEnabled(false);
		result_button.setEnabled(false);
		primary_text.setVisibility(View.INVISIBLE);
		gray_text.setVisibility(View.INVISIBLE);
		cpu_text.setVisibility(View.INVISIBLE);
		gpu_text.setVisibility(View.INVISIBLE);
		differ_text.setVisibility(View.INVISIBLE);


		Button.OnClickListener listener = new Button.OnClickListener() {
			public void onClick(View v) {
				switch (v.getId()) {
				case R.id.choose_button:
					Intent i = new Intent(
							Intent.ACTION_PICK,
							android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
					startActivityForResult(i, RESULT_LOAD_IMAGE);
					break;
				case R.id.sobel_gpu_cpu:
					Toast.makeText(
							ImageSobelFilter.this,
							"The image will save at:\n/storage/sdcard0/Sobel_input_gray.jpg\n/storage/sdcard0/cpu_sobel.jpg\n/storage/sdcard0/cpu_sobel.jpg\n/storage/sdcard0/differenceImg.jpg",
							Toast.LENGTH_SHORT).show();
					result_button.setEnabled(true);
					t = new nativeSobelFilter(imagePathString);
					t.start();
					break;
				case R.id.result_button:
					resultString = t.resultString;
					Toast.makeText(ImageSobelFilter.this, resultString,
							Toast.LENGTH_LONG).show();

					primary_text.setVisibility(View.VISIBLE);
					gray_text.setVisibility(View.VISIBLE);
					cpu_text.setVisibility(View.VISIBLE);
					gpu_text.setVisibility(View.VISIBLE);
					differ_text.setVisibility(View.VISIBLE);

					sobel_gpu_image.setImageBitmap(handleOOM(
							"/storage/sdcard0/gpu_sobel.jpg", sobel_gpu_image));
					sobel_cpu_image.setImageBitmap(handleOOM(
							"/storage/sdcard0/cpu_sobel.jpg", sobel_cpu_image));
					sobel_gray_image.setImageBitmap(handleOOM(
							"/storage/sdcard0/Sobel_input_gray.jpg",
							sobel_gray_image));
					sobel_differ_image.setImageBitmap(handleOOM(
							"/storage/sdcard0/differenceImg.jpg",
							sobel_differ_image));
					break;
				default:
				}
			}
		};
		chooseButton.setOnClickListener(listener);
		sobelGPU_CPU.setOnClickListener(listener);
		result_button.setOnClickListener(listener);

	}

	@Override
	protected void onStart() {
		super.onStart();
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
			if (checkSelfPermission(READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
				requestPermissions(
						new String[]{"android.permission.READ_EXTERNAL_STORAGE",
								"android.permission.READ_EXTERNAL_STORAGE"}, 0);
			}
		}
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		getMenuInflater().inflate(R.menu.image_sobel_filter, menu);
		return true;
	}

	protected void onActivityResult(int requestCode, int resultCode, Intent data) {
		super.onActivityResult(requestCode, resultCode, data);

		if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK
				&& null != data) {
			Uri selectedImage = data.getData();
			String[] filePathColumn = { MediaStore.Images.Media.DATA };
			Cursor cursor = getContentResolver().query(selectedImage,
					filePathColumn, null, null, null);
			cursor.moveToFirst();
			int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
			imagePathString = cursor.getString(columnIndex);
			cursor.close();
			sobelGPU_CPU.setEnabled(true);
			primaryImageView.setImageBitmap(handleOOM(imagePathString,
					primaryImageView));
		}
	}

	private Bitmap handleOOM(String absolutePath, ImageView imageView) {
		Bitmap bm;
		BitmapFactory.Options opt = new BitmapFactory.Options();
		opt.inJustDecodeBounds = true;
		bm = BitmapFactory.decodeFile(absolutePath, opt);

		int picWidth = opt.outWidth;
		int picHeight = opt.outHeight;

		int width = imageView.getWidth();
		int height = imageView.getHeight();
		opt.inSampleSize = 1;

		if (picWidth > picHeight) {
			if (picWidth > width)
				opt.inSampleSize = picWidth / width;
		} else {
			if (picHeight > height)
				opt.inSampleSize = picHeight / height;
		}
		opt.inJustDecodeBounds = false;
		bm = BitmapFactory.decodeFile(absolutePath, opt);
		return bm;
	}
}

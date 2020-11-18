package com.example.tflitemodelbenchmarking;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.util.Log;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

import org.tensorflow.lite.Interpreter;


public class MainActivity extends AppCompatActivity {
  private static final String TAG = "LowPoweredML_Benchmark";
  private static final int NUM_IMAGES = 100;
  private static final String XRAY_DIR = "xrayImages";
  private final String MODEL_ASSETS_PATH = "chexnet_pruned_model.tflite";
  private static int toastDuration = Toast.LENGTH_SHORT;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    Toolbar toolbar = findViewById(R.id.toolbar);
    setSupportActionBar(toolbar);
  }

  public void run_chexnet_pruned_quant(View view) throws IOException {
    runTest("chexnet_pruned_quant.tflite");
  }

  public void run_chexnet_pruned_model(View view) throws IOException {
    runTest("chexnet_pruned_model.tflite");
  }

  public void run_chexnet_quant2(View view) throws IOException {
    runTest("chexnet_quant2.tflite");
  }

  public void runTest(String modelName) throws IOException {
    Context context = getApplicationContext();
    float startTime = System.nanoTime();
    float testTime = 0.0f;
    float fastestInference = Float.MAX_VALUE;
    float slowestInference = Float.MIN_VALUE;

    String[] xrayList = getAssets().list(XRAY_DIR);
    if (xrayList != null) {
      // Run test on all images in directory
      Toast toast = Toast.makeText(context, "Running test. This can take a few minutes.", toastDuration);
      int counter = 1;

      for (String xray : xrayList) {
        toast.show();
        Log.d(TAG, "Running inference on image: " + xray);

        // Pass image into model, and time the inference
        float runTime = timeInference(modelName, xray);
        testTime += runTime;
        fastestInference = Math.min(fastestInference, runTime);
        slowestInference = Math.max(slowestInference, runTime);

        // Update user
        toast = Toast.makeText(context, "Running test. Completed " + counter++ + " / " + NUM_IMAGES, toastDuration);
        toast.show();
      }

      // Log stats
      float endTime = System.nanoTime();
      Log.d(TAG, "Completed test of " + modelName + " in " + (endTime - startTime) + " ns.");
      Log.d(TAG, "Average inference time per image: " + (testTime / NUM_IMAGES));
      Log.d(TAG, "Fastest inference: " + fastestInference + " ns.");
      Log.d(TAG, "Slowest inference: " + slowestInference + " ns.");
      toast = Toast.makeText(context,
                                    "Test completed; avg latency: " + (testTime / NUM_IMAGES),
                                    Toast.LENGTH_LONG);
      toast.show();
    } else {
      Log.d(TAG, "Directory name is not valid");
      Toast toast = Toast.makeText(context, "Error: No tests were run. See logs", Toast.LENGTH_LONG);
      toast.show();
    }
  }

  private float timeInference(String modelName, String image) throws IOException {
    Context context = getApplicationContext();
    float[][][][] input = loadImageAsFloatArr(image);
    float[][] output = new float[1][14];
    float time = System.nanoTime();

    Interpreter interpreter = new Interpreter(loadTfliteModel(modelName));
    Log.d(TAG, "Model loaded: " + MODEL_ASSETS_PATH);
    interpreter.run(input, output);
    float runTime = System.nanoTime() - time;
    interpreter.close();

    // Logging and toasts
    CharSequence toastText = "Inference completed in: " + runTime
      + " ns; Result: " + getIndexOfLargest(output[0]);
    Toast toast = Toast.makeText(context, toastText, toastDuration);
    toast.show();
    Log.d(TAG, "Output: " + Arrays.toString(output[0]));
    Log.d(TAG, "Likeliest index: " + getIndexOfLargest(output[0]));
    Log.d(TAG, "Inference done in " + runTime + " ns.");

    return runTime;
  }

  private float[][][][] loadImageAsFloatArr(String fileName) {
    Log.d(TAG, "Loading fileName: " + fileName);
    Bitmap bitmap = getBitmapFromAsset(getApplicationContext(), XRAY_DIR + "/" + fileName);
    Log.d(TAG, "File loaded: " + bitmap.getWidth() + "x" + bitmap.getHeight());

    bitmap = resizeBitmap(bitmap, 224, 224);
    Log.d(TAG, "Resized bitmap: " + bitmap.getWidth() + "x" + bitmap.getHeight());

    int[] intArray = new int[bitmap.getWidth() * bitmap.getHeight()];
    // Get all pixels and store in int array
    bitmap.getPixels(intArray, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    float[][][][] floatArray = new float[1][224][224][3];
    // Convert hexadecimal int array to RGB float array
    for (int i = 0; i < intArray.length; i++) {
      int pixel = intArray[i];
      float red = Color.red(pixel) / 255.0F;
      float green = Color.green(pixel) / 255.0F;
      float blue = Color.blue(pixel) / 255.0F;

      int xIndex = convert1Dto2D_x(i, bitmap.getWidth());
      int yIndex = convert1Dto2D_y(i, bitmap.getWidth());
      floatArray[0][xIndex][yIndex][0] = red;
      floatArray[0][xIndex][yIndex][1] = green;
      floatArray[0][xIndex][yIndex][2] = blue;
    }
    return floatArray;
  }

  private static int convert1Dto2D_x(int pix1DIndex, int width) {
    return pix1DIndex / width;
  }

  private static int convert1Dto2D_y(int pix1DIndex, int width) {
    return pix1DIndex % width;
  }

  public static Bitmap getBitmapFromAsset(Context context, String filePath) {
    AssetManager assetManager = context.getAssets();
    InputStream istr;
    Bitmap bitmap = null;
    try {
      istr = assetManager.open(filePath);
      bitmap = BitmapFactory.decodeStream(istr);
    } catch (IOException e) {
      Log.e(TAG, "Could not decode stream from " + filePath);
    }
    return bitmap;
  }

  private MappedByteBuffer loadTfliteModel(String modelName) throws IOException {
    Context context = getApplicationContext();
    AssetFileDescriptor assetFileDescriptor = context.getAssets().openFd(modelName);
    FileInputStream fileInputStream =
      new FileInputStream(assetFileDescriptor.getFileDescriptor());
    FileChannel fileChannel = fileInputStream.getChannel();
    long startoffset = assetFileDescriptor.getStartOffset();
    long declaredLength = assetFileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength);
  }

  public Bitmap resizeBitmap(Bitmap bm, int newHeight, int newWidth) {
    int width = bm.getWidth();
    int height = bm.getHeight();
    float scaleWidth = ((float) newWidth) / width;
    float scaleHeight = ((float) newHeight) / height;

    // Create a matrix for the manipulation
    Matrix matrix = new Matrix();

    // Resize the bit map
    matrix.postScale(scaleWidth, scaleHeight);

    // Recreate the new Bitmap
    return Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false);
  }

  @Override
  public boolean onCreateOptionsMenu(Menu menu) {
    // Inflate the menu; this adds items to the action bar if it is present.
    getMenuInflater().inflate(R.menu.menu_main, menu);
    return true;
  }

  /**
   * Since the model returns a float[14] array, get the index with the highest probability
   * diagnosis.
   **/
  public int getIndexOfLargest(float[] array) {
    if (array == null || array.length == 0) {
      return -1;
    }
    int largest = 0;
    for (int i = 1; i < array.length; i++) {
      if (array[i] > array[largest]) {
        largest = i;
      }
    }
    return largest; // if duplicates, position of the first largest found
  }

  @Override
  public boolean onOptionsItemSelected(MenuItem item) {
    // Handle action bar item clicks here. The action bar will
    // automatically handle clicks on the Home/Up button, so long
    // as you specify a parent activity in AndroidManifest.xml.
    int id = item.getItemId();

    //noinspection SimplifiableIfStatement
    if (id == R.id.action_settings) {
      return true;
    }

    return super.onOptionsItemSelected(item);
  }
}

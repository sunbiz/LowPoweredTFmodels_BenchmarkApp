package com.example.runmodelsj;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.media.Image;
import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

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
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

import org.tensorflow.lite.Interpreter;


public class MainActivity extends AppCompatActivity {
  private static final String TAG = "LowPoweredML_Benchmark";
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
    runModel("chexnet_pruned_quant.tflite");
  }

  public void run_chexnet_pruned_model(View view) throws IOException {
    runModel("chexnet_pruned_model.tflite");
  }

  public void runModel(String modelName) throws IOException {
    Context context = getApplicationContext();

    float[][][][] input = loadImage("00000001_000.png");
    float[][] output = new float[1][14];
    float time = System.nanoTime();
    try (Interpreter interpreter =
           new Interpreter(loadModelFile(modelName))) {
      Log.d(TAG, "Model loaded: " + MODEL_ASSETS_PATH);
      interpreter.run(input, output);
      float runTime = System.nanoTime() - time;

      CharSequence toastText = "Inference completed in: " + runTime
                                  + " ns; Result: " + getIndexOfLargest(output[0]);
      Toast toast = Toast.makeText(context, toastText, toastDuration);
      toast.show();

      Log.d(TAG, "Output: " + Arrays.toString(output[0]));
      Log.d(TAG, "Likeliest index: " + getIndexOfLargest(output[0]));
      Log.d(TAG, "Inference done in " + runTime + " ns.");
    }
  }

  private float[][][][] loadImage(String fileName) throws IOException {
    Bitmap bitmap = getBitmapFromAsset(getApplicationContext(), fileName);
    Log.d(TAG, "File loaded: " + bitmap.getWidth() + "x" + bitmap.getHeight());

    bitmap = getResizedBitmap(bitmap, 224, 224);
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
//      Log.d(TAG, "red: " + red + " green: " + green + " blue: " + blue);

      int xIndex = convert1Dto2D_x(i, bitmap.getWidth());
      int yIndex = convert1Dto2D_y(i, bitmap.getWidth());
      floatArray[0][xIndex][yIndex][0] = red;
      floatArray[0][xIndex][yIndex][1] = green;
      floatArray[0][xIndex][yIndex][2] = blue;
    }
    return floatArray;
  }

  private static int convert1Dto2D_x (int pix1DIndex, int width) {
    return pix1DIndex / width;
  }

  private static int convert1Dto2D_y (int pix1DIndex, int width) {
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

  private MappedByteBuffer loadModelFile(String modelName) throws IOException {
    Context context = getApplicationContext();
    AssetFileDescriptor assetFileDescriptor = context.getAssets().openFd(modelName);
    FileInputStream fileInputStream =
      new FileInputStream(assetFileDescriptor.getFileDescriptor());
    FileChannel fileChannel = fileInputStream.getChannel();
    long startoffset = assetFileDescriptor.getStartOffset();
    long declaredLength = assetFileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength);
  }

  public Bitmap getResizedBitmap(Bitmap bm, int newHeight, int newWidth) {
    int width = bm.getWidth();
    int height = bm.getHeight();
    float scaleWidth = ((float) newWidth) / width;
    float scaleHeight = ((float) newHeight) / height;

    // Create a matrix for the manipulation
    Matrix matrix = new Matrix();

    // Resize the bit map
    matrix.postScale(scaleWidth, scaleHeight);

    // Recreate the new Bitmap
    Bitmap resizedBitmap = Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false);
    return resizedBitmap;

  }

  @Override
  public boolean onCreateOptionsMenu(Menu menu) {
    // Inflate the menu; this adds items to the action bar if it is present.
    getMenuInflater().inflate(R.menu.menu_main, menu);
    return true;
  }

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
    return largest; // position of the first largest found
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

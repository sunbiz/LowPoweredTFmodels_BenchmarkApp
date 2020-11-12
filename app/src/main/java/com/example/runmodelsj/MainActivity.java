package com.example.runmodelsj;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
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

    public void runPerformanceTest(View view) throws IOException {
      Context context = getApplicationContext();
      // Just a null input for now
      float[][][][] input = new float[1][224][224][3];
//      int[] input = loadImage("00000001_000.png");

      float[][] output = new float[1][14];
      float time = System.nanoTime();
      try (Interpreter interpreter =
             new Interpreter(loadModelFile("chexnet_pruned_quant.tflite"))) {
        Log.d(TAG, "Model loaded: " + MODEL_ASSETS_PATH);
        interpreter.run(input, output);
        interpreter.close();
        float runTime = System.nanoTime() - time;

        CharSequence toastText = "Inference completed in: " + runTime + " ns";
        Toast toast = Toast.makeText(context, toastText, toastDuration);
        toast.show();

        Log.d(TAG, "Inference done in " + runTime + " ns.");
        Log.d(TAG, "Output: " + Arrays.toString(output[0]));
        Log.d(TAG, "Likeliest index: " + getIndexOfLargest(output[0]));
      }
    }

    private int[] loadImage(String fileName) throws IOException {
      Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open(fileName));
      for (int i = 0; i < 10; i++) {
        Log.d(TAG, "sample int pixel: " + bitmap.getPixel(i, i));
      }
      bitmap = getResizedBitmap(bitmap, 224, 224);
      int x = bitmap.getWidth();
      int y = bitmap.getHeight();
      int[] intArray = new int[x * y];
      bitmap.getPixels(intArray, 0, x, 0, 0, 224, 224);
      for (int i = 0; i < 10; i++) {
        Log.d(TAG, "sample int pixel: " + intArray[i]);
      }
      return intArray;
    }

    private float[][][][] convertIntArrayToInputDims(int[] intArray) {
      // Input intArray should be int[224 * 224]
      float[][][][] floatArr = new float[1][224][224][3];
      for (int pixVal : intArray) {

      }
      return floatArr;
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

package com.example.tflitemodelbenchmarking;

import android.content.Context;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.os.PowerManager;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Toast;

import java.io.IOException;
import java.util.Arrays;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity {
  private static final String TAG = "TFliteModelBenchmarking";
  private static final String XRAY_DIR = "xrayImages";
  private static final int toastDuration = Toast.LENGTH_SHORT;
  private static PowerManager.WakeLock wakeLock;
  private static String[] labels = {"Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia"};

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    Toolbar toolbar = findViewById(R.id.toolbar);
    setSupportActionBar(toolbar);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
    wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,
      "Benchmarking::RunPerformanceTest");
  }

  public void run_all(View view) throws IOException {
    for (int i = 0; i < 10; i++) {

      runTest("chexBaselineModel.tflite"); // from bruce chou's repository

      runTest("chexnet_pruned_model.tflite");
      runTest("chexnet_pruned_quant.tflite");
      runTest("chexnet_quant.tflite");
      runTest("chexnet_quant2.tflite");

//       Below models throw error
//      runTest("chexnet_pruned_quant_f16.tflite");
//      runTest("chexnet_pruned_quant_io.tflite");
//      runTest("chexnet_quant_f16.tflite");
//      runTest("chexnet_rsna_quant_io.tflite");
//      runTest("chexnet_ts_quant_io.tflite");
    }
  }

  public void run_baseline(View view) throws IOException {
    runTest("chexBaselineModel.tflite");
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

  public void run_chexnet_quant(View view) throws IOException {
    runTest("chexnet_quant.tflite");
  }

  public void runTest(String modelName) throws IOException {
    wakeLock.acquire(10*60*1000L /*10 minutes*/);

    float startTime = System.nanoTime();
    float testTime = 0.0f;
    float minLatency = Float.MAX_VALUE;
    float maxLatency = Float.MIN_VALUE;

    String[] xrayList = getAssets().list(XRAY_DIR);
    int numImages = xrayList.length;
    if (numImages > 0) {
      // Run test on all images in directory
      for (String xray : xrayList) {
        Log.v(TAG, "Running inference on image: " + xray);
        // Pass image into model, time the inference, and close the model
        float runTime = timeInference(modelName, xray);
        testTime += runTime;
        minLatency = Math.min(minLatency, runTime);
        maxLatency = Math.max(maxLatency, runTime);
      }

      // Log stats
      float endTime = System.nanoTime();
      Log.d(TAG, "Completed test of " + modelName + " in " + (endTime - startTime) + " ns on " + numImages + " images.");
      Log.d(TAG, "Average inference time per image: " + (testTime / numImages));
      Log.d(TAG, "Min latency: " + minLatency + " ns.");
      Log.d(TAG, "Max latency: " + maxLatency + " ns.");
    } else {
      Log.e(TAG, "No files found. Tests not run.");
    }
    wakeLock.release();
  }

  private float timeInference(String modelName, String image) throws IOException {
    float[][][][] input = Utils.loadImageAsFloatArr(getApplicationContext(), image);
    float[][] output = new float[1][14];

    Interpreter interpreter =
      new Interpreter(Utils.loadTfliteModel(getApplicationContext(), modelName));
    Log.v(TAG, "Model loaded: " + modelName);
    float time = System.nanoTime();
    interpreter.run(input, output);
    float runTime = System.nanoTime() - time;
    interpreter.close();

    // Logging and toasts
    CharSequence toastText = "Inference completed in: " + runTime
      + " ns; Result: " + Utils.getIndexOfLargest(output[0]);
//    Toast toast = Toast.makeText(context, toastText, toastDuration);
//    toast.show();
    Log.v(TAG, "Output: " + Arrays.toString(output[0]));
    Log.v(TAG, "Likeliest index: " + Utils.getIndexOfLargest(output[0]));
    Log.v(TAG, "Likeliest label: " + labels[Utils.getIndexOfLargest(output[0])]);
    Log.v(TAG, "Inference done in " + runTime + " ns.");

    return runTime;
  }

  @Override
  public boolean onCreateOptionsMenu(Menu menu) {
    // Inflate the menu; this adds items to the action bar if it is present.
    getMenuInflater().inflate(R.menu.menu_main, menu);
    return true;
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

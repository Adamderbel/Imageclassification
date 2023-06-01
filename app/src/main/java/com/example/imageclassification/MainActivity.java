package com.example.imageclassification;


import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;



import com.example.imageclassification.ml.Mobmodel;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button takepic,selectpic;
    int imageSize = 32;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        takepic= findViewById(R.id.go);
        selectpic=findViewById(R.id.button2);

        selectpic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraintent = new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraintent, 1);
            }
        });
        takepic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }
    @Override
    public void onActivityResult(int requestCode, int resultCode,  Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
            if (requestCode == 3) {
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getHeight(), image.getWidth());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }else{
                Uri dat=data.getData();
                Bitmap image =null;
                try{
                    image= MediaStore.Images.Media.getBitmap(this.getContentResolver(),dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
    public void classifyImage(Bitmap image){
        try {
            Mobmodel model = Mobmodel.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1,32 ,32, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer= ByteBuffer.allocateDirect(4* imageSize*imageSize*3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int [] intValue= new int [imageSize*imageSize];
            image.getPixels(intValue,0,image.getWidth(),0,0,image.getWidth(),image.getHeight());
            int pixel =0 ;
            for(int i =0;i<imageSize;i++){
                for(int j=0;j<imageSize;j++){
                    int val= intValue[pixel++];
                    byteBuffer.putFloat(((val>>16) & 0xFF) *(1.f/1));
                    byteBuffer.putFloat(((val>>8) & 0xFF) *(1.f/1));
                    byteBuffer.putFloat(((val & 0xFF) & 0xFF) *(1.f/1));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Mobmodel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] confidences =outputFeature0.getFloatArray();
            int maxPos=0;
            float maxConfidence=0;
            for(int i=0;i< confidences.length;i++){
                if(confidences[i]>maxConfidence){
                    maxConfidence=confidences[i];
                    maxPos=i;
                }
            }
            String[] classes={"apple","corn","kiwi","lemon","tomato"};
            result.setText(classes[maxPos]);
            String s="";
            for(int i=0; i< classes.length;i++){
                s+=String.format("%s: %.1f%%\n",classes[i],confidences[i]*100);
            }
            confidence.setText(s);
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

}
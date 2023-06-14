package com.example.ml

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.getSystemService
import com.example.ml.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {
    
    lateinit var btn: Button
    lateinit var sbtn: Button //gallery
    lateinit var pbtn: Button // predict
    lateinit var textView: TextView
    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        sbtn = findViewById(R.id.sbtn)
        pbtn = findViewById(R.id.pbtn)
        textView = findViewById(R.id.textView)
        imageView = findViewById(R.id.imageView)
        btn = findViewById(R.id.btn)

        //switching to camera activity
        btn.setOnClickListener {
            var Intent = Intent(this,CameraActivity::class.java)
            startActivity(Intent)
        }

        //importing the labels.txt file
        var labels = application.assets.open("labels.txt").bufferedReader().readLines()

        //image processor
        var imageProcessor = ImageProcessor.Builder()
            //.add(NormalizeOp(0.0f,255.0f))
            .add(ResizeOp(224,224,ResizeOp.ResizeMethod.BILINEAR))
            .build()

        sbtn.setOnClickListener {
            var intent = Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(intent, 100)
        }

        pbtn.setOnClickListener {

            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)

            tensorImage = imageProcessor.process(tensorImage)

            val model = Model.newInstance(this)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(tensorImage.buffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            var maxIdx = 0
            outputFeature0.forEachIndexed { index, fl ->
                if (outputFeature0[maxIdx] < fl){
                    maxIdx = index
                }
            }

            textView.setText(labels[maxIdx])

            // Releases model resources if no longer used.
            model.close()
        }

    }

    // Outside the main function
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        /*if (requestCode == 101){
            var pic = data?.getParcelableExtra<Bitmap>("data")
            imageView.setImageBitmap(pic)
        }*/

        if(requestCode == 100){
            var uri = data?.data;
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imageView.setImageBitmap(bitmap)
        }
    }
}
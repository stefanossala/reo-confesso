package com.reoconfesso.diabetesrisk

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Minimal screen for the synthetic diabetes-risk demo.
 * Inputs are plain text fields to keep the layout small; the point of the
 * project is the model inside the app, not the UI.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var model: DiabetesModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        model = DiabetesModel(this)

        val age = findViewById<EditText>(R.id.age)
        val bmi = findViewById<EditText>(R.id.bmi)
        val activity = findViewById<EditText>(R.id.activity)
        val bp = findViewById<EditText>(R.id.bp)
        val glucose = findViewById<EditText>(R.id.glucose)
        val sex = findViewById<EditText>(R.id.sex)
        val smoker = findViewById<EditText>(R.id.smoker)
        val family = findViewById<EditText>(R.id.family)
        val zone = findViewById<EditText>(R.id.zone)
        val result = findViewById<TextView>(R.id.result)

        findViewById<Button>(R.id.computeButton).setOnClickListener {
            try {
                val p = model.predict(
                    age = age.text.toString().toFloat(),
                    bmi = bmi.text.toString().toFloat(),
                    activityH = activity.text.toString().toFloat(),
                    systolicBp = bp.text.toString().toFloat(),
                    glucose = glucose.text.toString().toFloat(),
                    sexIsMale = sex.text.toString().trim().uppercase() == "M",
                    smoker = smoker.text.toString().trim() == "1",
                    familyHistory = family.text.toString().trim() == "1",
                    zone = zone.text.toString().trim(),
                )
                result.text = "5-year risk: %.1f%%".format(p * 100)
            } catch (e: Exception) {
                result.text = "Check your inputs: ${e.message}"
            }
        }
    }
}

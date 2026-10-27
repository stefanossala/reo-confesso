package com.reoconfesso.diabetesrisk

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Loads the on-device diabetes-risk model (model.tflite) and the preprocessing
 * parameters (preprocessing.json) from the app assets, and runs inference.
 *
 * The app MUST reproduce exactly the preprocessing used at training time:
 * same feature order, same (value - mean) / std normalisation. Those numbers
 * come from preprocessing.json so they can never drift from the Python side.
 *
 * This is a synthetic-data research demo (project "REo confesso").
 */
class DiabetesModel(context: Context) {

    private val interpreter: Interpreter
    private val featureOrder: List<String>
    private val mean: FloatArray
    private val std: FloatArray

    init {
        // 1. Load the TFLite model from assets into a memory-mapped buffer.
        interpreter = Interpreter(loadModelFile(context, "model.tflite"))

        // 2. Load preprocessing parameters (feature order + normalisation).
        val json = JSONObject(
            context.assets.open("preprocessing.json").bufferedReader().use { it.readText() }
        )
        featureOrder = json.getJSONArray("feature_order").let { arr ->
            List(arr.length()) { arr.getString(it) }
        }
        mean = json.getJSONArray("mean").toFloatArray()
        std = json.getJSONArray("std").toFloatArray()
    }

    /**
     * Raw inputs as the user types them. Returns the probability (0..1)
     * of type 2 diabetes onset within 5 years.
     */
    fun predict(
        age: Float,
        bmi: Float,
        activityH: Float,
        systolicBp: Float,
        glucose: Float,
        sexIsMale: Boolean,
        smoker: Boolean,
        familyHistory: Boolean,
        zone: String,          // "A".."E"
    ): Float {
        // Build one raw value per feature name, in the exact training order.
        val raw = FloatArray(featureOrder.size) { i ->
            when (val name = featureOrder[i]) {
                "age" -> age
                "bmi" -> bmi
                "activity_h" -> activityH
                "systolic_bp" -> systolicBp
                "glucose" -> glucose
                "sex_M" -> if (sexIsMale) 1f else 0f
                "smoker" -> if (smoker) 1f else 0f
                "family_history" -> if (familyHistory) 1f else 0f
                else -> when {
                    name.startsWith("zone_") ->
                        if (name == "zone_${zone.uppercase()}") 1f else 0f
                    else -> throw IllegalStateException("Unknown feature: $name")
                }
            }
        }

        // Normalise with the training mean/std.
        val input = ByteBuffer.allocateDirect(4 * raw.size).order(ByteOrder.nativeOrder())
        for (i in raw.indices) {
            input.putFloat((raw[i] - mean[i]) / std[i])
        }
        input.rewind()

        // Run inference: output is a single sigmoid probability.
        val output = Array(1) { FloatArray(1) }
        interpreter.run(input, output)
        return output[0][0]
    }

    private fun loadModelFile(context: Context, name: String): MappedByteBuffer {
        val fd = context.assets.openFd(name)
        FileInputStream(fd.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
        }
    }

    private fun org.json.JSONArray.toFloatArray() =
        FloatArray(length()) { getDouble(it).toFloat() }
}

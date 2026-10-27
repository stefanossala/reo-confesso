package com.reoconfesso.diabetesrisk

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * App B (obfuscated build): the model is NOT shipped as a plain file.
 * assets/model.enc is AES-256-CBC encrypted; it is decrypted in memory at
 * startup with a key hard-coded below, then handed to the interpreter.
 *
 * NOTE: hard-coding the key inside the app is exactly the (bad) practice this
 * project critiques. An attacker who reads the decompiled code recovers the key
 * and the scheme, so encryption + a hidden key is NOT real anonymisation --
 * it only raises the effort. That is the point we measure in phase 4.
 */
class DiabetesModel(context: Context) {

    private val interpreter: Interpreter
    private val featureOrder: List<String>
    private val mean: FloatArray
    private val std: FloatArray

    init {
        val encrypted = context.assets.open("model.enc").use { it.readBytes() }
        val modelBytes = decrypt(encrypted)
        val buffer = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
        buffer.put(modelBytes)
        buffer.rewind()
        interpreter = Interpreter(buffer)

        val json = JSONObject(
            context.assets.open("preprocessing.json").bufferedReader().use { it.readText() }
        )
        featureOrder = json.getJSONArray("feature_order").let { arr ->
            List(arr.length()) { arr.getString(it) }
        }
        mean = json.getJSONArray("mean").toFloatArray()
        std = json.getJSONArray("std").toFloatArray()
    }

    fun predict(
        age: Float,
        bmi: Float,
        activityH: Float,
        systolicBp: Float,
        glucose: Float,
        sexIsMale: Boolean,
        smoker: Boolean,
        familyHistory: Boolean,
        zone: String,
    ): Float {
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
                else -> if (name == "zone_${zone.uppercase()}") 1f else 0f
            }
        }
        val input = ByteBuffer.allocateDirect(4 * raw.size).order(ByteOrder.nativeOrder())
        for (i in raw.indices) input.putFloat((raw[i] - mean[i]) / std[i])
        input.rewind()
        val output = Array(1) { FloatArray(1) }
        interpreter.run(input, output)
        return output[0][0]
    }

    private fun decrypt(enc: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("AES/CBC/PKCS5Padding")
        cipher.init(
            Cipher.DECRYPT_MODE,
            SecretKeySpec(hexToBytes(KEY_HEX), "AES"),
            IvParameterSpec(hexToBytes(IV_HEX)),
        )
        return cipher.doFinal(enc)
    }

    private fun hexToBytes(h: String) = ByteArray(h.length / 2) {
        ((Character.digit(h[it * 2], 16) shl 4) + Character.digit(h[it * 2 + 1], 16)).toByte()
    }

    private fun JSONArray.toFloatArray() = FloatArray(length()) { getDouble(it).toFloat() }

    companion object {
        private const val KEY_HEX = "5c983dd89768cdb41a604c09cd68c45b0cb7631d29c06eb64bedf0913c0cfde9"
        private const val IV_HEX = "cf7356f3439c52c7e4a0b810b94f8d3b"
    }
}

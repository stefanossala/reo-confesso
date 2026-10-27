plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.reoconfesso.diabetesrisk"

    androidResources {
        noCompress += "tflite"
    }
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.reoconfesso.diabetesriskb"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            optimization {
                enable = true
                packageScope = setOf("androidx.**", "kotlin.**", "kotlinx.**")
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

kotlin {
    compilerOptions {
        // LiteRT 2.3.0 is built with a newer Kotlin than AGP 9's bundled compiler.
        // We only use its Java Interpreter class, so skip the metadata-version check.
        freeCompilerArgs.add("-Xskip-metadata-version-check")
    }
}

dependencies {
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)
    implementation("com.google.ai.edge.litert:litert:2.3.0")
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}
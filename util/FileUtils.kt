package com.sessizduygular.util

import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File

object FileUtils {
    fun createMultipart(file: File, fieldName: String): MultipartBody.Part {
        val reqFile = file.asRequestBody("image/*".toMediaTypeOrNull())
        return MultipartBody.Part.createFormData(fieldName, file.name, reqFile)
    }
}

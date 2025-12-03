#include <nvimgcodec.h>

void *NvimgcodecLoadSymbol(const char *name);

#define LOAD_SYMBOL_FUNC Nvimgcodec##LoadSymbol

#pragma GCC diagnostic ignored "-Wattributes"


nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecGetPropertiesNotFound(nvimgcodecProperties_t *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecGetProperties(nvimgcodecProperties_t * properties) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecProperties_t *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecGetProperties")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecGetProperties")) :
                           nvimgcodecGetPropertiesNotFound;
  return func_ptr(properties);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecInstanceCreateNotFound(nvimgcodecInstance_t *, const nvimgcodecInstanceCreateInfo_t *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecInstanceCreate(nvimgcodecInstance_t * instance, const nvimgcodecInstanceCreateInfo_t * create_info) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecInstance_t *, const nvimgcodecInstanceCreateInfo_t *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecInstanceCreate")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecInstanceCreate")) :
                           nvimgcodecInstanceCreateNotFound;
  return func_ptr(instance, create_info);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecInstanceDestroyNotFound(nvimgcodecInstance_t) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecInstanceDestroy(nvimgcodecInstance_t instance) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecInstance_t);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecInstanceDestroy")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecInstanceDestroy")) :
                           nvimgcodecInstanceDestroyNotFound;
  return func_ptr(instance);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecExtensionCreateNotFound(nvimgcodecInstance_t, nvimgcodecExtension_t *, nvimgcodecExtensionDesc_t *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecExtensionCreate(nvimgcodecInstance_t instance, nvimgcodecExtension_t * extension, nvimgcodecExtensionDesc_t * extension_desc) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecInstance_t, nvimgcodecExtension_t *, nvimgcodecExtensionDesc_t *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecExtensionCreate")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecExtensionCreate")) :
                           nvimgcodecExtensionCreateNotFound;
  return func_ptr(instance, extension, extension_desc);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecExtensionDestroyNotFound(nvimgcodecExtension_t) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecExtensionDestroy(nvimgcodecExtension_t extension) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecExtension_t);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecExtensionDestroy")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecExtensionDestroy")) :
                           nvimgcodecExtensionDestroyNotFound;
  return func_ptr(extension);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecFutureWaitForAllNotFound(nvimgcodecFuture_t) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecFutureWaitForAll(nvimgcodecFuture_t future) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecFuture_t);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecFutureWaitForAll")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecFutureWaitForAll")) :
                           nvimgcodecFutureWaitForAllNotFound;
  return func_ptr(future);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecFutureDestroyNotFound(nvimgcodecFuture_t) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecFutureDestroy(nvimgcodecFuture_t future) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecFuture_t);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecFutureDestroy")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecFutureDestroy")) :
                           nvimgcodecFutureDestroyNotFound;
  return func_ptr(future);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecFutureGetProcessingStatusNotFound(nvimgcodecFuture_t, nvimgcodecProcessingStatus_t *, size_t *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecFutureGetProcessingStatus(nvimgcodecFuture_t future, nvimgcodecProcessingStatus_t * processing_status, size_t * size) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecFuture_t, nvimgcodecProcessingStatus_t *, size_t *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecFutureGetProcessingStatus")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecFutureGetProcessingStatus")) :
                           nvimgcodecFutureGetProcessingStatusNotFound;
  return func_ptr(future, processing_status, size);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecImageCreateNotFound(nvimgcodecInstance_t, nvimgcodecImage_t *, const nvimgcodecImageInfo_t *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecImageCreate(nvimgcodecInstance_t instance, nvimgcodecImage_t * image, const nvimgcodecImageInfo_t * image_info) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecInstance_t, nvimgcodecImage_t *, const nvimgcodecImageInfo_t *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecImageCreate")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecImageCreate")) :
                           nvimgcodecImageCreateNotFound;
  return func_ptr(instance, image, image_info);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecImageDestroyNotFound(nvimgcodecImage_t) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecImageDestroy(nvimgcodecImage_t image) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecImage_t);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecImageDestroy")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecImageDestroy")) :
                           nvimgcodecImageDestroyNotFound;
  return func_ptr(image);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecCodeStreamCreateFromFileNotFound(nvimgcodecInstance_t, nvimgcodecCodeStream_t *, const char *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecCodeStreamCreateFromFile(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t * code_stream, const char * file_name) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecInstance_t, nvimgcodecCodeStream_t *, const char *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamCreateFromFile")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamCreateFromFile")) :
                           nvimgcodecCodeStreamCreateFromFileNotFound;
  return func_ptr(instance, code_stream, file_name);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecCodeStreamCreateFromHostMemNotFound(nvimgcodecInstance_t, nvimgcodecCodeStream_t *, const unsigned char *, size_t) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecCodeStreamCreateFromHostMem(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t * code_stream, const unsigned char * data, size_t length) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecInstance_t, nvimgcodecCodeStream_t *, const unsigned char *, size_t);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamCreateFromHostMem")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamCreateFromHostMem")) :
                           nvimgcodecCodeStreamCreateFromHostMemNotFound;
  return func_ptr(instance, code_stream, data, length);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecCodeStreamDestroyNotFound(nvimgcodecCodeStream_t) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecCodeStreamDestroy(nvimgcodecCodeStream_t code_stream) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecCodeStream_t);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamDestroy")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamDestroy")) :
                           nvimgcodecCodeStreamDestroyNotFound;
  return func_ptr(code_stream);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecCodeStreamGetCodeStreamInfoNotFound(nvimgcodecCodeStream_t, nvimgcodecCodeStreamInfo_t *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecCodeStreamGetCodeStreamInfo(nvimgcodecCodeStream_t code_stream, nvimgcodecCodeStreamInfo_t * codestream_info) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecCodeStream_t, nvimgcodecCodeStreamInfo_t *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamGetCodeStreamInfo")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamGetCodeStreamInfo")) :
                           nvimgcodecCodeStreamGetCodeStreamInfoNotFound;
  return func_ptr(code_stream, codestream_info);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecCodeStreamGetSubCodeStreamNotFound(nvimgcodecCodeStream_t, nvimgcodecCodeStream_t *, const nvimgcodecCodeStreamView_t *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecCodeStreamGetSubCodeStream(nvimgcodecCodeStream_t code_stream, nvimgcodecCodeStream_t * sub_code_stream, const nvimgcodecCodeStreamView_t * code_stream_view) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecCodeStream_t, nvimgcodecCodeStream_t *, const nvimgcodecCodeStreamView_t *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamGetSubCodeStream")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamGetSubCodeStream")) :
                           nvimgcodecCodeStreamGetSubCodeStreamNotFound;
  return func_ptr(code_stream, sub_code_stream, code_stream_view);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecCodeStreamGetImageInfoNotFound(nvimgcodecCodeStream_t, nvimgcodecImageInfo_t *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecCodeStreamGetImageInfo(nvimgcodecCodeStream_t code_stream, nvimgcodecImageInfo_t * image_info) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecCodeStream_t, nvimgcodecImageInfo_t *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamGetImageInfo")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecCodeStreamGetImageInfo")) :
                           nvimgcodecCodeStreamGetImageInfoNotFound;
  return func_ptr(code_stream, image_info);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecDecoderCreateNotFound(nvimgcodecInstance_t, nvimgcodecDecoder_t *, const nvimgcodecExecutionParams_t *, const char *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecDecoderCreate(nvimgcodecInstance_t instance, nvimgcodecDecoder_t * decoder, const nvimgcodecExecutionParams_t * exec_params, const char * options) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecInstance_t, nvimgcodecDecoder_t *, const nvimgcodecExecutionParams_t *, const char *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecDecoderCreate")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecDecoderCreate")) :
                           nvimgcodecDecoderCreateNotFound;
  return func_ptr(instance, decoder, exec_params, options);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecDecoderDestroyNotFound(nvimgcodecDecoder_t) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecDecoderDestroy(nvimgcodecDecoder_t decoder) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecDecoder_t);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecDecoderDestroy")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecDecoderDestroy")) :
                           nvimgcodecDecoderDestroyNotFound;
  return func_ptr(decoder);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecDecoderGetMetadataNotFound(nvimgcodecDecoder_t, nvimgcodecCodeStream_t, nvimgcodecMetadata_t **, int *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecDecoderGetMetadata(nvimgcodecDecoder_t decoder, nvimgcodecCodeStream_t code_stream, nvimgcodecMetadata_t ** metadata, int * metadata_count) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecDecoder_t, nvimgcodecCodeStream_t, nvimgcodecMetadata_t **, int *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecDecoderGetMetadata")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecDecoderGetMetadata")) :
                           nvimgcodecDecoderGetMetadataNotFound;
  return func_ptr(decoder, code_stream, metadata, metadata_count);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecDecoderCanDecodeNotFound(nvimgcodecDecoder_t, const nvimgcodecCodeStream_t *, const nvimgcodecImage_t *, int, const nvimgcodecDecodeParams_t *, nvimgcodecProcessingStatus_t *, int) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecDecoderCanDecode(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStream_t * streams, const nvimgcodecImage_t * images, int batch_size, const nvimgcodecDecodeParams_t * params, nvimgcodecProcessingStatus_t * processing_status, int force_format) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecDecoder_t, const nvimgcodecCodeStream_t *, const nvimgcodecImage_t *, int, const nvimgcodecDecodeParams_t *, nvimgcodecProcessingStatus_t *, int);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecDecoderCanDecode")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecDecoderCanDecode")) :
                           nvimgcodecDecoderCanDecodeNotFound;
  return func_ptr(decoder, streams, images, batch_size, params, processing_status, force_format);
}

nvimgcodecStatus_t NVIMGCODECAPI nvimgcodecDecoderDecodeNotFound(nvimgcodecDecoder_t, const nvimgcodecCodeStream_t *, const nvimgcodecImage_t *, int, const nvimgcodecDecodeParams_t *, nvimgcodecFuture_t *) {
  return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecStatus_t nvimgcodecDecoderDecode(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStream_t * streams, const nvimgcodecImage_t * images, int batch_size, const nvimgcodecDecodeParams_t * params, nvimgcodecFuture_t * future) {
  using FuncPtr = nvimgcodecStatus_t (NVIMGCODECAPI *)(nvimgcodecDecoder_t, const nvimgcodecCodeStream_t *, const nvimgcodecImage_t *, int, const nvimgcodecDecodeParams_t *, nvimgcodecFuture_t *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecDecoderDecode")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvimgcodecDecoderDecode")) :
                           nvimgcodecDecoderDecodeNotFound;
  return func_ptr(decoder, streams, images, batch_size, params, future);
}

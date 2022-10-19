#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <typeinfo>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

int main(int argc, char* argv[]) {

    // ------------------------------- Init model ------------------------------- //

    std::string instanceName{"BiSeNet-ONNX-inference"};
    std::string modelFilepath{"src/bisenet.onnx"};
    std::string imageFilepath{"src/images/79.jpg"};
    const int64_t batchSize = 1;
    bool useCUDA{true};
    const char *useCUDAFlag = "--use_cuda";
    const char *useCPUFlag = "--use_cpu";
    if (argc == 1) {
        useCUDA = false;
    } else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0)) {
        useCUDA = true;
    } else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0)) {
        useCUDA = false;
    } else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) != 0)) {
        useCUDA = false;
    } else {
        throw std::runtime_error{"Too many arguments."};
    }

    if (useCUDA) {
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
    } else {
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    if (useCUDA) {
        // Using CUDA backend
        // https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    const char *inputName = session.GetInputName(0, allocator);

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    if (inputDims.at(0) == -1) {
//        std::cout << "Got dynamic batch size. Setting input batch size to "
//                  << batchSize << "." << std::endl;
        inputDims.at(0) = batchSize;
    }

    const char *outputName = session.GetOutputName(0, allocator);

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    if (outputDims.at(0) == -1) {
//        std::cout << "Got dynamic batch size. Setting output batch size to "
//                  << batchSize << "." << std::endl;
        outputDims.at(0) = batchSize;
    }

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
    std::cout << "Input Name: " << inputName << std::endl;
    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input Dimensions: " << inputDims << std::endl;
    std::cout << "Output Name: " << outputName << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output Dimensions: " << outputDims << std::endl;

    // ------------------------------- Pre-processing ------------------------------- //

    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR,
               cv::Size(inputDims.at(3), inputDims.at(2)),
               cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resizedImage);

    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    // ------------------------------- Inference ONNX ------------------------------- //
    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);

    // Make copies of the same image input.
    for (int64_t i = 0; i < batchSize; ++i) {
        std::copy(preprocessedImage.begin<float>(),
                  preprocessedImage.end<float>(),
                  inputTensorValues.begin() + i * inputTensorSize / batchSize);
    }

    size_t outputTensorSize = vectorProduct(outputDims);
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<const char *> inputNames{inputName};
    std::vector<const char *> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
            inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensorValues.data(), outputTensorSize,
            outputDims.data(), outputDims.size()));


    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1 /*Number of inputs*/, outputNames.data(),
                outputTensors.data(), 1 /*Number of outputs*/);

    // ------------------------------- Post-processing ------------------------------- //

    int outHeight = inputDims.at(2);
    int outWidth = inputDims.at(3);
    auto *pred = outputTensors.front().GetTensorMutableData<float>();
    std::vector<float> output_tensor_values(outHeight * outWidth);
    std::vector<float> output_tensor_values2(outHeight * outWidth);
    size_t t = outHeight * outWidth - 1;
    for (size_t i = 0; i < outHeight * outWidth * 2; i++)
    {
        if (i <= t)
        {
            output_tensor_values.at(i) = pred[i];
        }
        else
        {
            output_tensor_values2.at(i - (outHeight * outWidth)) = pred[i];
        }
    }

    double min_value, max_value;
    cv::Mat result(outHeight, outWidth, CV_32FC1, output_tensor_values.data());
    minMaxLoc(result, &min_value, &max_value, nullptr, nullptr);
    result = (result - min_value) / (max_value - min_value);
    result *= 255;
    result.convertTo(result, CV_8UC1);

    /*
     * THRESH đóng vai trò như confident score
     * THRESH càng cao thì mask càng sát
     */
    cv::threshold(result, result, 220, 255, cv::THRESH_BINARY);

    // Rollback size
    cv::resize(result, result, cv::Size(imageBGR.size[1], imageBGR.size[0]), cv::INTER_CUBIC);

    // Visualize
    cv::Mat in[] = {result / 10, result, result};
    cv::merge(in, 3, result);

    double alpha = 0.5; double beta; double input;
    cv::Mat dst;
    beta = ( 1.0 - alpha );
    addWeighted( result, alpha, imageBGR, beta, 0.0, dst);

    bool show_image = true;
    if (show_image){
        cv::imshow("Visualization", dst);
        cv::waitKey(0);
    }

    // Measure latency
    bool MEASURE = false;
    if (MEASURE)
    {
        std::cout << "Calculating measure latency...\n" << std::endl;
        int numTests{100};
        std::chrono::steady_clock::time_point begin =
                std::chrono::steady_clock::now();
        for (int i = 0; i < numTests; i++) {
            session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                        inputTensors.data(), 1, outputNames.data(),
                        outputTensors.data(), 1);
        }
        std::chrono::steady_clock::time_point end =
                std::chrono::steady_clock::now();
        std::cout << "Minimum Inference Latency: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                           begin)
                             .count() /
                     static_cast<float>(numTests)
                  << " ms" << std::endl;
    }

}
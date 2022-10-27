#include "inference.h"

cv::Mat run(int argc, char* argv[])
{
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
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

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
        inputDims.at(0) = batchSize;
    }

    const char *outputName = session.GetOutputName(0, allocator);

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    if (outputDims.at(0) == -1) {
        outputDims.at(0) = batchSize;
    }

//    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
//    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
//    std::cout << "Input Name: " << inputName << std::endl;
//    std::cout << "Input Type: " << inputType << std::endl;
//    std::cout << "Input Dimensions: " << inputDims << std::endl;
//    std::cout << "Output Name: " << outputName << std::endl;
//    std::cout << "Output Type: " << outputType << std::endl;
//    std::cout << "Output Dimensions: " << outputDims << std::endl;

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
     * THRESH đóng vai trò như confidence score
     * THRESH càng cao thì mask càng sát
     */
    cv::threshold(result, result, 220, 255, cv::THRESH_BINARY);

    // Rollback size
    cv::resize(result, result, cv::Size(imageBGR.size[1], imageBGR.size[0]), cv::INTER_CUBIC);

    return result;
}
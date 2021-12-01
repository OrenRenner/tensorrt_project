#include "mmpose_tensorrt.hpp"

class Logger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const char* msg) noexcept override;
} gLogger;


tensorrt::MmposeTensorRT::MmposeTensorRT(const std::string& model, const std::string device)
{
    this->model = model;
    this->device = device;
    this->batch_size = 1;
    this->context = nullptr;
    this->engine = nullptr;
}

tensorrt::MmposeTensorRT::~MmposeTensorRT()
{
}

bool tensorrt::MmposeTensorRT::initialize(const void* userdata)
{
    return this->parseOnnxModel(this->model, this->engine, this->context);;
}

void* tensorrt::MmposeTensorRT::calculate(const cv::Mat& mat, size_t& count)
{
    std::vector<void*> buffers_tmp(this->engine->getNbBindings()); // buffers for input and output data
    this->buffers = buffers_tmp;

    for (size_t i = 0; i < this->engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(this->engine->getBindingDimensions(i)) * this->batch_size * sizeof(float);
        cudaMalloc(&this->buffers[i], binding_size);
        if (this->engine->bindingIsInput(i))
        {
            this->input_dims.emplace_back(this->engine->getBindingDimensions(i));
        }
        else
        {
            this->output_dims.emplace_back(this->engine->getBindingDimensions(i));
        }
    }
    if (this->input_dims.empty() || this->output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
        return nullptr;
    }

    cv::Mat frame;
    mat.copyTo(frame);
    // preprocess input data
    preprocessImage(frame, (float*)this->buffers[0], this->input_dims[0]);

    // inference
    context->enqueue(this->batch_size, this->buffers.data(), 0, nullptr);

    // postprocess results
    std::vector<cv::Point> points = postprocessResults((float*)buffers[1], output_dims[0], batch_size, frame.size());

    cv::Point* result = new cv::Point[points.size()];
    for (int i = 0; i < points.size(); i++) result[i] = points[i];
    count = points.size();
    return static_cast<void*>(result);
}

bool tensorrt::MmposeTensorRT::parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine, TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{
    auto networkFlags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::IBuilder> builder{ nvinfer1::createInferBuilder(gLogger) };
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{ builder->createNetworkV2(networkFlags) };  // >createNetwork()
    TRTUniquePtr<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*network, gLogger) };
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{ builder->createBuilderConfig() };
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return false;
    }
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch
    builder->setMaxBatchSize(100);
    // generate TensorRT engine optimized for the target platform
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());

    return true;
}

void tensorrt::MmposeTensorRT::preprocessImage(cv::Mat& image, float* gpu_input, const nvinfer1::Dims& dims)
{
    // read input image
    cv::Mat frame;
    image.copyTo(frame);
    if (frame.empty())
    {
        std::cerr << "Input image load failed\n";
        return;
    }
    cv::cvtColor(frame, frame,
        cv::ColorConversionCodes::COLOR_BGR2RGB);

    cv::cuda::GpuMat gpu_frame;
    // upload image to GPU
    gpu_frame.upload(frame);

    auto input_width = dims.d[3];
    auto input_height = dims.d[2];
    auto channels = 3; // dims.d[0];
    auto input_size = cv::Size(input_width, input_height);  //input_width, input_height
    // resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    // normalize
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    // to tensor
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
}

std::vector<cv::Point> tensorrt::MmposeTensorRT::postprocessResults(float* gpu_output, const nvinfer1::Dims& dims, int batch_size, cv::Size origin_size)
{
    // copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<cv::Point> points;

    int i = 0;
    int x_coor = 0, y_coor = 0;

    while (true) {
        if (i % 2 == 0) x_coor = ceil(cpu_output[i] * origin_size.width);
        else y_coor = ceil(cpu_output[i] * origin_size.height);

        if (y_coor == 0 && x_coor == 0) break;

        i++;
        if (y_coor != 0 && x_coor != 0) {
            cv::Point tmp(x_coor, y_coor);
            points.push_back(tmp);
            x_coor = 0; y_coor = 0;
        }
    }

    return points;
}

void Logger::log(Severity severity, const char* msg) noexcept
{
    // suppress info-level messages
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}


// calculate size of tensor
size_t tensorrt::MmposeTensorRT::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

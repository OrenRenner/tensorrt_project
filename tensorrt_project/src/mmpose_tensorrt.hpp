#pragma once

#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>

namespace tensorrt {

	// destroy TensorRT objects if something goes wrong
	struct TRTDestroy
	{
		template <class T>
		void operator()(T* obj) const
		{
			if (obj)
			{
				obj->destroy();
			}
		}
	};

	template <class T>
	using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

	class MmposeTensorRT {
	public:
		MmposeTensorRT(const std::string& model, const std::string device = "GPU");
		~MmposeTensorRT();

		bool initialize(const void* userdata = nullptr);
		void* calculate(const cv::Mat& mat, size_t& count);

	private:
		std::string model;
		std::string device;
		int batch_size;

		// initialize TensorRT engine and parse ONNX model
		TRTUniquePtr<nvinfer1::ICudaEngine> engine;
		TRTUniquePtr<nvinfer1::IExecutionContext> context;

		// get sizes of input and output and allocate memory required for input data and for output data
		std::vector<nvinfer1::Dims> input_dims; // we expect only one input
		std::vector<nvinfer1::Dims> output_dims; // and one output

		std::vector<void*> buffers; // buffers for input and output data

		// calculate size of tensor
		size_t getSizeByDim(const nvinfer1::Dims& dims);
		bool parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
			TRTUniquePtr<nvinfer1::IExecutionContext>& context);
		void preprocessImage(cv::Mat& image, float* gpu_input, const nvinfer1::Dims& dims);
		std::vector<cv::Point> postprocessResults(float* gpu_output, const nvinfer1::Dims& dims, int batch_size, cv::Size origin_size);
	};


};

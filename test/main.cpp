#include"../tensorrt_project/src/mmpose_tensorrt.hpp"
#include<iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>


int main() {
	// Creation model
	tensorrt::MmposeTensorRT skeleton("../mmpose_keypoint.onnx");
	std::vector<cv::Mat> images;
	//Initialize model
	if (skeleton.initialize()) {
		images.push_back(cv::imread("../images/test1.jpg"));
		images.push_back(cv::imread("../images/test2.jpg"));
		images.push_back(cv::imread("../images/test3.jpg"));

		for (int i = 0; i < 3; i++) {
			cv::Mat tmp_image = images[i];

			// Run model
			size_t count = 0;
			cv::Point* result = static_cast<cv::Point*>(skeleton.calculate(tmp_image, count));

			/*cv::resize(tmp_image, tmp_image,
				cv::Size(256, 256),
				cv::InterpolationFlags::INTER_CUBIC);*/

				// Draw result
			for (int i = 0; i < count; i++) {
				cv::circle(tmp_image, result[i], 2, cv::Scalar(255, 0, 0));
			}

			// Show result
			cv::namedWindow("image", 0);
			cv::imshow("image", tmp_image);
			cv::waitKey(0);
		}
	}
	return 0;
}
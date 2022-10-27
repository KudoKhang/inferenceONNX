#include "inference.cpp"

int main(int argc, char* argv[]) {
    cv::Mat result;
    result = run(argc, argv);
    cv::imshow("result", result);
    cv::waitKey(0);
}

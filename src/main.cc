/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <fstream>
#include <json/json.h>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include "yoloxp.h"

class InputParser
{
  public:
    InputParser(int &argc, char **argv)
    {
        for (int i = 1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
    }
    std::string getCmdOption(const std::string &option) const
    {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end())
        {
            return *itr;
        }
        static std::string empty_string("");
        return empty_string;
    }
    bool cmdOptionExists(const std::string &option) const
    {
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }

  private:
    std::vector<std::string> tokens;
};

int main(int argc, char **argv)
{
    InputParser input(argc, argv);
    if (input.cmdOptionExists("-h"))
    {
        printf("Usage 1: ./validate_coco --engine path_to_engine_or_loadable  --coco_path path_to_coco_dataset "
               "--backend cudla_fp16/cudla_int8\n");
        printf("Usage 2: ./validate_coco --engine path_to_engine_or_loadable  --image path_to_image --backend "
               "cudla_fp16/cudla_int8\n");
        return 0;
    }
    std::string engine_path = input.getCmdOption("--engine");
    if (engine_path.empty())
    {
        printf("Error: please specify the loadable path with --engine");
        return 0;
    }
    std::string backend_str = input.getCmdOption("--backend");
    std::string coco_path   = input.getCmdOption("--coco_path");
    std::string image_path  = input.getCmdOption("--image");

    YoloxpBackend backend = YoloxpBackend::CUDLA_FP16;
    if (backend_str == "cudla_int8")
    {
        backend = YoloxpBackend::CUDLA_INT8;
    }

    yoloxp yoloxp_infer(engine_path, backend);

    std::vector<cv::Mat>            bgr_imgs;
    std::vector<std::vector<float>> results;

    if (!image_path.empty())
    {
        printf("Run Yoloxp DLA pipeline for %s\n", image_path.c_str());
        cv::Mat image = cv::imread(image_path);
        bgr_imgs.push_back(image);
        yoloxp_infer.preProcess4Validate(bgr_imgs);

        yoloxp_infer.infer();
        results = yoloxp_infer.postProcess4Validation();
        printf("Num object detect: %ld\n", results.size());

        // cv::Mat dst_image;
        // const float input_height = 960.0;
        // const float input_width = 960.0;
        // const float scale = std::min(input_width / image.cols, input_height / image.rows);

        // const auto scale_size = cv::Size(image.cols * scale, image.rows * scale);
        // cv::resize(image, dst_image, scale_size, 0, 0, cv::INTER_CUBIC);
        // const auto bottom = input_height - dst_image.rows;
        // const auto right = input_width - dst_image.cols;
        // copyMakeBorder(
        // dst_image, dst_image, 0, bottom, 0, right, cv::BORDER_CONSTANT, {114, 114, 114});


        for (auto &item : results)
        {
            printf("score: %lf,  left: %lf , top: %lf , right: %lf , bottom: %lf\n", item[5], item[0], item[1], item[2], item[3]);
            // left, top, right, bottom, label, confident
            cv::rectangle(image, cv::Point(item[0], item[1]), cv::Point(item[2], item[3]), cv::Scalar(0, 255, 0), 2,
                          16);
        }
        printf("detect result has been write to result.jpg\n");
        cv::imwrite("result.jpg", image);
    }

    return 0;
}
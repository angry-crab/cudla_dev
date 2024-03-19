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
#include <filesystem>

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

template <typename ... Args>
std::string format(const std::string& fmt, Args ... args )
{
  size_t len = std::snprintf( nullptr, 0, fmt.c_str(), args ... );
  std::vector<char> buf(len + 1);
  std::snprintf(&buf[0], len + 1, fmt.c_str(), args ... );
  return std::string(&buf[0], &buf[0] + len);
}

std::string getFilename(std::string& file_path)
{
    int pos_dot = file_path.find_last_of(".");
    int pos_slash = file_path.find_last_of("/");
    std::string name = file_path.substr(pos_slash+1, pos_dot-pos_slash-1);
    return name;
}

void saveBoxPred(std::vector<std::string>& map2class, std::vector<std::vector<float>>& boxes, std::string path, std::string target_path)
{
    std::string file_path = path;
    // std::cout << "get file: " << filename << std::endl;
    std::string body = getFilename(file_path);
    std::string out = target_path + "/" + body + ".txt";
    // std::cout << "save to: " << out << std::endl;
    std::ofstream ofs;
    ofs.open(out, std::ios::out);
    // ofs.setf(std::ios::fixed, std::ios::floatfield);
    // ofs.precision(5);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
        //   ofs << box[4] << " "; // label
        //   ofs << box[5] << " "; // score
        //   ofs << box[0] << " "; // left
        //   ofs << box[1] << " "; // top
        //   ofs << box[2] << " "; // right
        //   ofs << box[3] << " "; // bottom
        std::string text = format("%s %f %d %d %d %d", map2class[box[4]].c_str(), box[5], (int)box[0], (int)box[1], (int)box[2], (int)box[3]);
        ofs << text << "\n";
        }
    }
    else {
    std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    return;
}

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

    std::vector<std::vector<int>> color_map{{255,255,255}, {0,0,255}, {0,160,165}, {100,0,200},
                                        {128,255,0}, {255,255,0}, {255,0,32}, {255,0,0}};
    std::vector<std::string> map2class{"UNKNOWN", "CAR", "TRUCK", "BUS", "BICYCLE", "MOTORBIKE", "PEDESTRIAN", "ANIMAL"};

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
        std::string target_path("/home/autoware/develop/cudla_dev/data/detection-results");
        std::filesystem::create_directory(target_path);
        for (const auto & entry : std::filesystem::directory_iterator(image_path))
        {
            std::string file = entry.path().string();
            printf("Run Yoloxp DLA pipeline for %s\n", file.c_str());
            cv::Mat image = cv::imread(file);
            bgr_imgs.push_back(image);
            yoloxp_infer.preProcess4Validate(bgr_imgs);

            yoloxp_infer.infer();
            results = yoloxp_infer.postProcess4Validation(image.cols, image.rows);
            printf("Num object detect: %ld\n", results.size());

            for (auto &item : results)
            {
                char buff[128];
                std::vector<int> rgb = color_map[item[4]];
                sprintf(buff, "%s %2.0f%%", map2class[item[4]].c_str(), item[5] * 100);
                // printf("score: %lf,  left: %lf , top: %lf , right: %lf , bottom: %lf\n", item[5], item[0], item[1], item[2], item[3]);
                // left, top, right, bottom, label, confident
                cv::rectangle(image, cv::Point(item[0], item[1]), cv::Point(item[2], item[3]), cv::Scalar(rgb[2], rgb[1], rgb[0]), 2,
                            16);
                cv::putText(image, buff, cv::Point(item[0], item[1]), 0, 1, cv::Scalar(rgb[2], rgb[1], rgb[0]), 2);
            }
            // printf("detect result has been write to result.jpg\n");
            std::string filename = getFilename(file);
            cv::imwrite("/home/autoware/develop/cudla_dev/data/results/" + filename + ".jpg", image);

            saveBoxPred(map2class, results, file, target_path);
            bgr_imgs.clear();
            results.clear();
        }
    }

    return 0;
}
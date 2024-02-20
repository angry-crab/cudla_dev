#include "yoloxp.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

static void hwc_to_chw(cv::InputArray src, cv::OutputArray dst)
{
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    // Stretch one-channel images to vector
    for (auto &img : channels)
    {
        img = img.reshape(1, 1);
    }
    // Concatenate three vectors to one
    cv::hconcat(channels, dst);
}

float limit(float a, float low, float high) { return a < low ? low : (a > high ? high : a); }

static float iou(const Box &a, const Box &b)
{
    float cleft   = std::max(a.left, b.left);
    float ctop    = std::max(a.top, b.top);
    float cright  = std::min(a.right, b.right);
    float cbottom = std::min(a.bottom, b.bottom);

    float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top);
    float b_area = std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top);
    return c_area / (a_area + b_area - c_area);
}

static BoxArray cpu_nms(BoxArray &boxes, float threshold)
{

    std::sort(boxes.begin(), boxes.end(),
              [](BoxArray::const_reference a, BoxArray::const_reference b) { return a.confidence > b.confidence; });

    BoxArray output;
    output.reserve(boxes.size());

    std::vector<bool> remove_flags(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i)
    {

        if (remove_flags[i])
            continue;

        auto &a = boxes[i];
        output.emplace_back(a);

        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (remove_flags[j])
                continue;

            auto &b = boxes[j];
            if (b.class_label == a.class_label)
            {
                if (iou(a, b) >= threshold)
                    remove_flags[j] = true;
            }
        }
    }
    return output;
}


yoloxp::yoloxp(std::string engine_path, YoloxpBackend backend)
{
    mEnginePath = engine_path;
    mBackend    = backend;

    output_h_.reserve(output_dims_reshape[0] * output_dims_reshape[1] * output_dims_reshape[2]);


    checkCudaErrors(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));

    mImgPushed = 0;
    // Malloc cuda memory for binding
    void *input_buf = nullptr;
    void *output_buf_0;
    void *output_buf_1;
    void *output_buf_2;
    // For FP16 and INT8
    if (mBackend == YoloxpBackend::CUDLA_FP16 || mBackend == YoloxpBackend::CUDLA_INT8)
    {
        // Create a cuda context first, otherwise cudla context creation would fail.
        cudaFree(0);
#ifdef USE_DLA_STANDALONE_MODE
        mCuDLACtx = new cuDLAContextStandalone(engine_path.c_str());
#else
        mCuDLACtx = new cuDLAContextHybrid(engine_path.c_str());
#endif
        checkCudaErrors(cudaMalloc(&mInputTemp1, 1 * 3 * 960 * 960 * sizeof(float)));
        if (mBackend == YoloxpBackend::CUDLA_FP16)
        {
            // Same size as cuDLA input
            checkCudaErrors(cudaMalloc(&mInputTemp2, mCuDLACtx->getInputTensorSizeWithIndex(0)));
        }
        if (mBackend == YoloxpBackend::CUDLA_INT8)
        {
            // For int8, we need to do reformat on FP32, then cast to INT8, so we need 4x space.
            checkCudaErrors(cudaMalloc(&mInputTemp2, 4 * mCuDLACtx->getInputTensorSizeWithIndex(0)));
        }
#ifdef USE_DLA_STANDALONE_MODE
        input_buf    = mCuDLACtx->getInputCudaBufferPtr(0);
        output_buf_0 = mCuDLACtx->getOutputCudaBufferPtr(0);
        output_buf_1 = mCuDLACtx->getOutputCudaBufferPtr(1);
        output_buf_2 = mCuDLACtx->getOutputCudaBufferPtr(2);
#else
        // Use fp16:chw16 output due to no int8 scale for the output tensor
        checkCudaErrors(cudaMalloc(&input_buf, mCuDLACtx->getInputTensorSizeWithIndex(0)));
        checkCudaErrors(cudaMalloc(&output_buf_0, mCuDLACtx->getOutputTensorSizeWithIndex(0)));
        checkCudaErrors(cudaMalloc(&output_buf_1, mCuDLACtx->getOutputTensorSizeWithIndex(1)));
        checkCudaErrors(cudaMalloc(&output_buf_2, mCuDLACtx->getOutputTensorSizeWithIndex(2)));

        std::vector<void *> cudla_inputs{input_buf};
        std::vector<void *> cudla_outputs{output_buf_0, output_buf_1, output_buf_2};
        mCuDLACtx->initTask(cudla_inputs, cudla_outputs);
#endif
    }
    mBindingArray.push_back(reinterpret_cast<void *>(input_buf));
    mBindingArray.push_back(output_buf_0);
    mBindingArray.push_back(output_buf_1);
    mBindingArray.push_back(output_buf_2);

    src = {mBindingArray[1], mBindingArray[2], mBindingArray[3]};
    
    // cudaMalloc((void **)&dst[0], sizeof(half) * 3 * 85 * 9261);
    // dst[1] = reinterpret_cast<half *>(dst[0]) + 3 * 85 * 7056;
    // dst[2] = reinterpret_cast<half *>(dst[1]) + 3 * 85 * 1764;

    // mReformatRunner = new ReformatRunner();

    checkCudaErrors(cudaMalloc(&mAffineMatrix, 3 * sizeof(float)));

    checkCudaErrors(cudaEventCreateWithFlags(&mStartEvent, cudaEventBlockingSync));
    checkCudaErrors(cudaEventCreateWithFlags(&mEndEvent, cudaEventBlockingSync));

}

yoloxp::~yoloxp()
{
    delete mCuDLACtx;
    mCuDLACtx = nullptr;
}

std::vector<cv::Mat> yoloxp::preProcess4Validate(std::vector<cv::Mat> &cv_img)
{
    const double norm_factor_ = 1.0;
    const auto batch_size = cv_img.size();
    const float input_height = static_cast<float>(input_dims[2]);
    const float input_width = static_cast<float>(input_dims[3]);
    std::vector<cv::Mat> dst_images;
    std::vector<float> scales_;

    for (const auto & image : images) {
        cv::Mat dst_image;
        const float scale = std::min(input_width / image.cols, input_height / image.rows);
        scales_.emplace_back(scale);
        const auto scale_size = cv::Size(image.cols * scale, image.rows * scale);
        cv::resize(image, dst_image, scale_size, 0, 0, cv::INTER_CUBIC);
        const auto bottom = input_height - dst_image.rows;
        const auto right = input_width - dst_image.cols;
        copyMakeBorder(
        dst_image, dst_image, 0, bottom, 0, right, cv::BORDER_CONSTANT, {114, 114, 114});
        dst_images.emplace_back(dst_image);
    }

    const auto chw_images = cv::dnn::blobFromImages(
        dst_images, norm_factor, cv::Size(), cv::Scalar(), false, false, CV_32F);

    const auto data_length = chw_images.total();
    input_h_.clear();
    input_h_.reserve(data_length);
    const auto flat = chw_images.reshape(1, data_length);
    input_h_ = chw_images.isContinuous() ? flat : flat.clone();

    this->pushImg(input_h_.data(), 1, true);
}

int yoloxp::pushImg(void *imgBuffer, int numImg, bool fromCPU)
{
    int dim = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];

    if (mBackend == YoloxpBackend::CUDLA_FP16)
    {
        std::vector<__half> tmp_fp(dim);
        convert_float_to_half((float *)imgBuffer, (__half *)tmp_fp.data(), dim);

        checkCudaErrors(cudaMemcpy(mBindingArray.data(), tmp_fp.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
        // convert_float_to_half((float *)mInputTemp1, (__half *)mInputTemp2, 1 * 3 * 960 * 960);
        // std::vector<void *> vec_temp_2{mInputTemp2};
        // mReformatRunner->ReformatImage(vec_temp_2.data(), mBindingArray.data(), mStream);
    }
    if (mBackend == YoloxpBackend::CUDLA_INT8)
    {
        // checkCudaErrors(cudaMemcpy(mInputTemp1, imgBuffer, dim * sizeof(float), cudaMemcpyHostToDevice));
        // std::vector<void *> vec_temp_1{mInputTemp1};
        // std::vector<void *> vec_temp_2{mInputTemp2};
        // mReformatRunner->ReformatImageV2(vec_temp_1.data(), vec_temp_2.data(), mStream);
        // checkCudaErrors(cudaStreamSynchronize(mStream));
        // dla_hwc4 format, so 
        // convert_float_to_int8((float *)mInputTemp2, (int8_t *)mBindingArray[0], 1 * 4 * 960 * 960, mInputScale);
    }
    mImgPushed += numImg;
    return 0;
}

int yoloxp::infer()
{
    if (mImgPushed == 0)
    {
        std::cerr << " error: mImgPushed = " << mImgPushed << "  ,mImgPushed == 0!" << std::endl;
        return -1;
    }
#ifndef USE_DLA_STANDALONE_MODE
    checkCudaErrors(cudaEventRecord(mStartEvent, mStream));
#endif

    if (mBackend == YoloxpBackend::CUDLA_FP16 || mBackend == YoloxpBackend::CUDLA_INT8)
    {
        mCuDLACtx->submitDLATask(mStream);
    }

#ifndef USE_DLA_STANDALONE_MODE
    checkCudaErrors(cudaEventRecord(mEndEvent, mStream));
    checkCudaErrors(cudaEventSynchronize(mEndEvent));
    checkCudaErrors(cudaEventElapsedTime(&ms, mStartEvent, mEndEvent));
    std::cout << "Inference time: " << ms << " ms" << std::endl;
#endif

    output_h_.clear();

    if (mBackend == YoloxpBackend::CUDLA_FP16)
    {
        int dim_0 = output_dims_0[0] * output_dims_0[1] * output_dims_0[2];
        std::vector<float> fp_0_float(dim_0);
        copyHalf2Float(fp_0_float, 1);

        int dim_1 = output_dims_1[0] * output_dims_1[1] * output_dims_1[2];
        std::vector<float> fp_1_float(dim_1);
        copyHalf2Float(fp_0_float, 2);
        
        int dim_2 = output_dims_2[0] * output_dims_2[1] * output_dims_2[2];
        std::vector<float> fp_2_float(dim_2);
        copyHalf2Float(fp_0_float, 3);

        std::memcopy((void *)&output_h_[0], fp_0_float.data(), dim_0 * sizeof(float));
        std::memcopy((void *)&output_h_[dim_0], fp_1_float.data(), dim_1 * sizeof(float));
        std::memcopy((void *)&output_h_[dim_0+dim_1], fp_2_float.data(), dim_2 * sizeof(float));
        
        // mReformatRunner->Run(src.data(), dst.data(), mStream);
    }
    if (mBackend == YoloxpBackend::CUDLA_INT8)
    {
        // Use fp16:chw16 output here
        // mReformatRunner->Run(src.data(), dst.data(), mStream);
    }

    checkCudaErrors(cudaStreamSynchronize(mStream));
    mImgPushed = 0;
    return 0;
}

std::vector<std::vector<float>> yoloxp::postProcess4Validation(float confidence_threshold, float nms_threshold)
{

}

void yoloxp::copyHalf2Float(std::vector<float>& out_float, int binding_idx)
{
    int dim_0 = out_float.size();
    std::vector<__half> fp_0(dim_0);
    checkCudaErrors(cudaMemcpy(fp_0.data(), (void *)mBindingArray[binding_idx], dim_0 * sizeof(__half), cudaMemcpyDeviceToHost));
    convert_half_to_float((__half *)fp_0.data(), (float *)out_float.data(), dim_0);
}
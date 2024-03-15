#include "yoloxp.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdint.h>
#include <chrono>

// static void hwc_to_chw(cv::InputArray src, cv::OutputArray dst)
// {
//     std::vector<cv::Mat> channels;
//     cv::split(src, channels);
//     // Stretch one-channel images to vector
//     for (auto &img : channels)
//     {
//         img = img.reshape(1, 1);
//     }
//     // Concatenate three vectors to one
//     cv::hconcat(channels, dst);
// }

static void convert_float_to_half(float * a, __half * b, int size) {
    for(int i=0; i<size; ++i)
    {
        b[i] = __float2half(a[i]);
    }
}

static void convert_half_to_float(__half * a, float * b, int size) {
    for(int i=0; i<size; ++i)
    {
        b[i] = __half2float(a[i]);
    }
}

static void convert_float_to_int8(const float* a, int8_t* b, int size, float scale) {
    for(int idx=0; idx<size; ++idx)
    {
        float v = (a[idx] / scale);
        if(v < -128) v = -128;
        if(v > 127) v = 127;
        b[idx] = (int8_t)v;
    }
}

// concat dim[-1]
static bool concat(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, 
                    std::vector<int>& dim_A, std::vector<int>& dim_B, std::vector<int>& dim_C, 
                    std::vector<float>& output)
{
    assert(dim_A.size() == 4);
    assert(dim_B.size() == 4);
    assert(dim_C.size() == 4);
    std::size_t pa = 0, pb = 0, pc = 0;
    std::size_t da = dim_A[3]*dim_A[2], db = dim_B[3]*dim_B[2], dc = dim_C[3]*dim_C[2];
    int step = 0;
    for(std::size_t i=0; i<output.size(); i+= da+db+dc, pa+=da, pb+=db, pc+=dc)
    {
        if(i+da >= output.size() || i+da+db >= output.size() || pa >= A.size() || pb >= B.size() || pc >= C.size())
        {
            return false;
        }
        std::memcpy((void*)&output[i], (void*)&A[pa], da*sizeof(float));
        std::memcpy((void*)&output[i+da], (void*)&B[pb], db*sizeof(float));
        std::memcpy((void*)&output[i+da+db], (void*)&C[pc], dc*sizeof(float));
        step++;
    }

    // std::cout << "concat step : " << step << std::endl;

    return true;
}

static void transpose2d(std::vector<float>& input, std::vector<float>& output, std::vector<int>& dim)
{
    assert(dim.size() == 2);
    int la = dim[0], lb = dim[1];
    for(int i=0; i<dim[0]; ++i)
    {
        for(int j=0; j<dim[1]; ++j)
        {
            int idx_from = i*lb + j;
            int idx_to = j*la + i;
            output[idx_to] = input[idx_from];
        }
    }
}

std::size_t roundup(std::size_t n, int byte)
{
    int factor = 64 / byte;
    std::size_t res = ((n+factor)/factor) * factor;
    return res;
}

static void reformat(std::vector<float>& input, std::vector<float>& output, std::vector<int>& dim_i, int byte)
{
    std::size_t step_i = roundup(dim_i.back(), byte);
    std::size_t step_o = static_cast<std::size_t>(dim_i.back());
    // std::cout << "step_i : " << step_i << std::endl;
    for(std::size_t pi=0, po=0; pi<input.size(); pi+=step_i, po+=step_o)
    {
        std::memcpy((void*)&output[po], (void*)&input[pi], dim_i.back()*sizeof(float));
    }
}

static float intersectionArea(const Object & a, const Object & b)
{
    cv::Rect a_rect(a.x_offset, a.y_offset, a.width, a.height);
    cv::Rect b_rect(b.x_offset, b.y_offset, b.width, b.height);
    cv::Rect_<float> inter = a_rect & b_rect;
    return inter.area();
}

static bool pad(__half *src, __half *dst, std::size_t total , std::size_t pad_size)
{
    if(src == nullptr || dst == nullptr)
    {
       return false;
    }
    for(std::size_t i=0; i<total; ++i)
    {
        if(i >= total)
        {
            return false;
        }
        dst[i*pad_size] = src[i];
    }
    return true;
}

// static BoxArray cpu_nms(BoxArray &boxes, float threshold)
// {

//     std::sort(boxes.begin(), boxes.end(),
//               [](BoxArray::const_reference a, BoxArray::const_reference b) { return a.confidence > b.confidence; });

//     BoxArray output;
//     output.reserve(boxes.size());

//     std::vector<bool> remove_flags(boxes.size());
//     for (size_t i = 0; i < boxes.size(); ++i)
//     {

//         if (remove_flags[i])
//             continue;

//         auto &a = boxes[i];
//         output.emplace_back(a);

//         for (size_t j = i + 1; j < boxes.size(); ++j)
//         {
//             if (remove_flags[j])
//                 continue;

//             auto &b = boxes[j];
//             if (b.class_label == a.class_label)
//             {
//                 if (iou(a, b) >= threshold)
//                     remove_flags[j] = true;
//             }
//         }
//     }
//     return output;
// }


yoloxp::yoloxp(std::string engine_path, YoloxpBackend backend)
{
    mEnginePath = engine_path;
    mBackend    = backend;

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
        // checkCudaErrors(cudaMalloc(&mInputTemp1, 1 * 3 * 960 * 960 * sizeof(__half)));
        if (mBackend == YoloxpBackend::CUDLA_FP16)
        {
            // mByte = 2;
            // Same size as cuDLA input
            // checkCudaErrors(cudaMalloc(&mInputTemp2, mCuDLACtx->getInputTensorSizeWithIndex(0)));
        }
        if (mBackend == YoloxpBackend::CUDLA_INT8)
        {
            // mByte = 1;
            // For int8, we need to do reformat on FP32, then cast to INT8, so we need 4x space.
            // checkCudaErrors(cudaMalloc(&mInputTemp2, 4 * mCuDLACtx->getInputTensorSizeWithIndex(0)));
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

        std::cout << "dla_intput size : " << mCuDLACtx->getInputTensorSizeWithIndex(0) << std::endl;
        std::vector<void *> cudla_inputs{input_buf};
        std::vector<void *> cudla_outputs{output_buf_0, output_buf_1, output_buf_2};
        mCuDLACtx->initTask(cudla_inputs, cudla_outputs);
#endif
    }
    mBindingArray.push_back(reinterpret_cast<void *>(input_buf));
    mBindingArray.push_back(output_buf_0);
    mBindingArray.push_back(output_buf_1);
    mBindingArray.push_back(output_buf_2);

    // src = {mBindingArray[1], mBindingArray[2], mBindingArray[3]};
    
    // cudaMalloc((void **)&dst[0], sizeof(half) * 3 * 85 * 9261);
    // dst[1] = reinterpret_cast<half *>(dst[0]) + 3 * 85 * 7056;
    // dst[2] = reinterpret_cast<half *>(dst[1]) + 3 * 85 * 1764;

    // mReformatRunner = new ReformatRunner();

    // checkCudaErrors(cudaMalloc(&mAffineMatrix, 3 * sizeof(float)));

    checkCudaErrors(cudaEventCreateWithFlags(&mStartEvent, cudaEventBlockingSync));
    checkCudaErrors(cudaEventCreateWithFlags(&mEndEvent, cudaEventBlockingSync));

}

yoloxp::~yoloxp()
{
    delete mCuDLACtx;
    mCuDLACtx = nullptr;
    float sum = std::accumulate(time_.begin(), time_.end(), 0);
    std::cout << "avg infer time : " << sum/time_.size() << std::endl;
}

void yoloxp::preProcess4Validate(std::vector<cv::Mat> &cv_img)
{
    const double norm_factor_ = 1.0;
    const auto batch_size = cv_img.size();
    const float input_height = static_cast<float>(input_dims[2]);
    const float input_width = static_cast<float>(input_dims[3]);
    std::vector<cv::Mat> dst_images;
    scales_.clear();
    input_h_.clear();

    for (const auto & image : cv_img) {
        cv::Mat dst_image;
        const float scale = std::min(input_width / image.cols, input_height / image.rows);

        // std::cout << "scale : " << scale << std::endl;
        // std::cout << "size : " << image.cols * scale << " , " << image.rows * scale << std::endl;

        scales_.emplace_back(scale);
        const auto scale_size = cv::Size(image.cols * scale, image.rows * scale);
        cv::resize(image, dst_image, scale_size, 0, 0, cv::INTER_LINEAR);
        const auto bottom = input_height - dst_image.rows;
        const auto right = input_width - dst_image.cols;
        copyMakeBorder(
        dst_image, dst_image, 0, bottom, 0, right, cv::BORDER_CONSTANT, {114, 114, 114});
        dst_images.emplace_back(dst_image);
        // cv::imwrite("tmp_img.jpg", dst_image);
    }

    const auto chw_images = cv::dnn::blobFromImages(
        dst_images, norm_factor_, cv::Size(), cv::Scalar(), false, false, CV_32F);

    std::cout << "chw_image : " << chw_images.size[0] << " , " << chw_images.size[1] << " , "
                << chw_images.size[2] << " , " << chw_images.size[3]   << std::endl;
    const auto data_length = chw_images.total();
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
        convert_float_to_half((float *)imgBuffer, tmp_fp.data(), dim);
        checkCudaErrors(cudaMemcpy(mCuDLACtx->getInputCudaBufferPtr(0), (void *)tmp_fp.data(), dim * sizeof(__half), cudaMemcpyHostToDevice));

        // convert_float_to_half((float *)mInputTemp1, (__half *)mInputTemp2, 1 * 3 * 960 * 960);
        // std::vector<void *> vec_temp_2{mInputTemp2};
        // mReformatRunner->ReformatImage(vec_temp_2.data(), mBindingArray.data(), mStream);
    }
    if (mBackend == YoloxpBackend::CUDLA_INT8)
    {
        std::vector<int8_t> tmp_int(dim);
        convert_float_to_int8((float *)imgBuffer, tmp_int.data(), dim, mInputScale);
        checkCudaErrors(cudaMemcpy(mCuDLACtx->getInputCudaBufferPtr(0), (void *)tmp_int.data(), dim * sizeof(int8_t), cudaMemcpyHostToDevice));
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
    output_h_.clear();
    output_h_ = std::vector<float>(output_dims_reshape[0]*output_dims_reshape[1]*output_dims_reshape[2]);
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
        checkCudaErrors(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        mCuDLACtx->submitDLATask(mStream);
        checkCudaErrors(cudaDeviceSynchronize());
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        elapsed).count();
        time_.push_back(milliseconds);
    }

#ifndef USE_DLA_STANDALONE_MODE
    checkCudaErrors(cudaEventRecord(mEndEvent, mStream));
    checkCudaErrors(cudaEventSynchronize(mEndEvent));
    checkCudaErrors(cudaEventElapsedTime(&ms, mStartEvent, mEndEvent));
    std::cout << "Inference time: " << ms << " ms" << std::endl;
#endif

    // if (mBackend == YoloxpBackend::CUDLA_FP16)
    // {
    //     // mReformatRunner->Run(src.data(), dst.data(), mStream);
    // }
    // if (mBackend == YoloxpBackend::CUDLA_INT8)
    // {
    //     // Use fp16:chw16 output here
    //     // mReformatRunner->Run(src.data(), dst.data(), mStream);
    // }

    // iff output format is fp16
    int dim3_0 = output_dims_0[0] * output_dims_0[1] * output_dims_0[2];
    int r_0 = roundup(output_dims_0[3], mByte);
    std::vector<float> fp_0_float(dim3_0 * r_0);
    copyHalf2Float(fp_0_float, 0);
    std::vector<float> fp_0(dim3_0 * output_dims_0[3], 0);
    reformat(fp_0_float, fp_0, output_dims_0, mByte);

    // print_dla_addr((half *)mCuDLACtx->getOutputCudaBufferPtr(0), 25, dim_0, mStream);
    // checkCudaErrors(cudaStreamSynchronize(mStream));

    int dim3_1 = output_dims_1[0] * output_dims_1[1] * output_dims_1[2];
    int r_1 = roundup(output_dims_1[3], mByte);
    std::vector<float> fp_1_float(dim3_1 * r_1);
    copyHalf2Float(fp_1_float, 1);
    std::vector<float> fp_1(dim3_1 * output_dims_1[3], 0);
    reformat(fp_1_float, fp_1, output_dims_1, mByte);

    // print_dla_addr((half *)mCuDLACtx->getOutputCudaBufferPtr(1), 20, dim_0, mStream);
    // checkCudaErrors(cudaStreamSynchronize(mStream));
    
    int dim3_2 = output_dims_2[0] * output_dims_2[1] * output_dims_2[2];
    int r_2 = roundup(output_dims_2[3], mByte);
    std::vector<float> fp_2_float(dim3_2 * r_2);
    copyHalf2Float(fp_2_float, 2);
    std::vector<float> fp_2(dim3_2 * output_dims_2[3], 0);
    reformat(fp_2_float, fp_2, output_dims_2, mByte);

    // print_dla_addr((half *)mCuDLACtx->getOutputCudaBufferPtr(2), 20, dim_0, mStream);
    // checkCudaErrors(cudaStreamSynchronize(mStream));

    std::vector<float> output_tmp(output_h_.size());
    if(concat(fp_0, fp_1, fp_2, output_dims_0, output_dims_1, output_dims_2, output_tmp))
    {
        // std::cout << "concat ok" << std::endl;
    }

    std::vector<int> dim_reshape{output_dims_reshape[2], output_dims_reshape[1]}; // (13, 18900)
    transpose2d(output_tmp, output_h_, dim_reshape);

    checkCudaErrors(cudaStreamSynchronize(mStream));
    mImgPushed = 0;
    return 0;
}

std::vector<std::vector<float>> yoloxp::postProcess4Validation(int img_w, int img_h)
{
    img_height = img_h;
    img_width = img_w;
    std::vector<std::vector<float>> res;
    ObjectArray object_array;
    decodeOutputs((float *)output_h_.data(), object_array, scales_[0]);

    for (const auto & object : object_array)
    {
        const auto left = object.x_offset;
        const auto top = object.y_offset;
        const auto right = std::clamp(left + object.width, 0, img_width);
        const auto bottom = std::clamp(top + object.height, 0, img_height);
        res.push_back(std::vector<float>{left, top, right, bottom, object.type, object.score});
    }
    return res;
}

void yoloxp::copyHalf2Float(std::vector<float>& out_float, int binding_idx)
{
    int dim_0 = out_float.size();
    std::vector<__half> fp_0(dim_0);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(fp_0.data(), mCuDLACtx->getOutputCudaBufferPtr(binding_idx), dim_0 * sizeof(__half), cudaMemcpyDeviceToHost));
    convert_half_to_float((__half *)fp_0.data(), (float *)out_float.data(), dim_0);
}

void yoloxp::decodeOutputs(
    float * prob, ObjectArray & objects, float scale) const
{
    ObjectArray proposals;
    const float input_height = static_cast<float>(input_dims[2]);
    const float input_width = static_cast<float>(input_dims[3]);
    std::vector<GridAndStride> grid_strides;
    std::vector<int> output_strides_{8, 16, 32};

    generateGridsAndStride(input_width, input_height, output_strides_, grid_strides);
    generateYoloxProposals(grid_strides, prob, score_threshold_, proposals);

    qsortDescentInplace(proposals);

    std::cout << "proposals size " << proposals.size()<< std::endl;

    std::vector<int> picked;
    // cspell: ignore Bboxes
    nmsSortedBboxes(proposals, picked, nms_threshold_);

    int count = static_cast<int>(picked.size());
    objects.resize(count);
    float scale_x = input_width / static_cast<float>(img_width);
    float scale_y = input_height / static_cast<float>(img_height);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        float x0, y0, x1, y1;
        // adjust offset to original unpadded
        if (scale == -1.0) {
            x0 = (objects[i].x_offset) / scale_x;
            y0 = (objects[i].y_offset) / scale_y;
            x1 = (objects[i].x_offset + objects[i].width) / scale_x;
            y1 = (objects[i].y_offset + objects[i].height) / scale_y;
        } else {
            x0 = (objects[i].x_offset) / scale;
            y0 = (objects[i].y_offset) / scale;
            x1 = (objects[i].x_offset + objects[i].width) / scale;
            y1 = (objects[i].y_offset + objects[i].height) / scale;
        }
        // clip
        x0 = std::clamp(x0, 0.f, static_cast<float>(img_width - 1));
        y0 = std::clamp(y0, 0.f, static_cast<float>(img_height - 1));
        x1 = std::clamp(x1, 0.f, static_cast<float>(img_width - 1));
        y1 = std::clamp(y1, 0.f, static_cast<float>(img_height - 1));

        objects[i].x_offset = x0;
        objects[i].y_offset = y0;
        objects[i].width = x1 - x0;
        objects[i].height = y1 - y0;
    }
}

void yoloxp::generateGridsAndStride(
  const int target_w, const int target_h, const std::vector<int> & strides,
  std::vector<GridAndStride> & grid_strides) const
{
  for (auto stride : strides) {
    int num_grid_w = target_w / stride;
    int num_grid_h = target_h / stride;
    for (int g1 = 0; g1 < num_grid_h; g1++) {
      for (int g0 = 0; g0 < num_grid_w; g0++) {
        grid_strides.push_back(GridAndStride{g0, g1, stride});
      }
    }
  }
}

void yoloxp::generateYoloxProposals(
  std::vector<GridAndStride> grid_strides, float * feat_blob, float prob_threshold,
  ObjectArray & objects) const
{
  const int num_anchors = grid_strides.size();

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    const int grid0 = grid_strides[anchor_idx].grid0;
    const int grid1 = grid_strides[anchor_idx].grid1;
    const int stride = grid_strides[anchor_idx].stride;

    const int basic_pos = anchor_idx * (num_class_ + 5);

    // yolox/models/yolo_head.py decode logic
    // To apply this logic, YOLOX head must output raw value
    // (i.e., `decode_in_inference` should be False)
    float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
    float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
    // exp is complex for embedded processors
    // float w = exp(feat_blob[basic_pos + 2]) * stride;
    // float h = exp(feat_blob[basic_pos + 3]) * stride;
    // float x0 = x_center - w * 0.5f;
    // float y0 = y_center - h * 0.5f;

    float box_objectness = feat_blob[basic_pos + 4];
    for (int class_idx = 0; class_idx < num_class_; class_idx++) {
      float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
      float box_prob = box_objectness * box_cls_score;
      if (box_prob > prob_threshold) {
        Object obj;
        // On-demand applying for exp
        float w = exp(feat_blob[basic_pos + 2]) * stride;
        float h = exp(feat_blob[basic_pos + 3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;
        obj.x_offset = x0;
        obj.y_offset = y0;
        obj.height = h;
        obj.width = w;
        obj.type = class_idx;
        obj.score = box_prob;

        objects.push_back(obj);
      }
    }  // class loop
  }    // point anchor loop
}

void yoloxp::qsortDescentInplace(ObjectArray & face_objects, int left, int right) const
{
  int i = left;
  int j = right;
  float p = face_objects[(left + right) / 2].score;

  while (i <= j) {
    while (face_objects[i].score > p) {
      i++;
    }

    while (face_objects[j].score < p) {
      j--;
    }

    if (i <= j) {
      // swap
      std::swap(face_objects[i], face_objects[j]);

      i++;
      j--;
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (left < j) {
        qsortDescentInplace(face_objects, left, j);
      }
    }
#pragma omp section
    {
      if (i < right) {
        qsortDescentInplace(face_objects, i, right);
      }
    }
  }
}

void yoloxp::nmsSortedBboxes(
  const ObjectArray & face_objects, std::vector<int> & picked, float nms_threshold) const
{
  picked.clear();
  const int n = face_objects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    cv::Rect rect(
      face_objects[i].x_offset, face_objects[i].y_offset, face_objects[i].width,
      face_objects[i].height);
    areas[i] = rect.area();
  }

  for (int i = 0; i < n; i++) {
    const Object & a = face_objects[i];

    int keep = 1;
    for (int j = 0; j < static_cast<int>(picked.size()); j++) {
      const Object & b = face_objects[picked[j]];

      // intersection over union
      float inter_area = intersectionArea(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold) {
        keep = 0;
      }
    }

    if (keep) {
      picked.push_back(i);
    }
  }
}
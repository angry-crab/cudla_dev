#include "yoloxp.h"
#include <cuda_runtime.h>
#include <iostream>

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

struct HalfVec
{
    HalfVec(std::size_t size) : size_(size)
    {
        if(data == nullptr)
        {
            data = malloc(size_ * sizeof(__half));
        }
    }

    ~HalfVec()
    {
        free(data);
    }

    __half *get()
    {
        return (__half *)data;
    }

    void *data;
    std::size_t size_;
};

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

// concat dim[-1]
static bool concat(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, 
                    std::vector<int>& dim_A, std::vector<int>& dim_B, std::vector<int>& dim_C, 
                    std::vector<float>& output)
{
    std::size_t pa = 0, pb = 0, pc = 0;
    std::size_t da = dim_A.back(), db = dim_B.back(), dc = dim_C.back();
    for(std::size_t i=0; i<output.size(); i+= da+db+dc, pa+=da, pb+=db, pc+=dc)
    {
        if(i+da >= output.size() || i+da+db >= output.size() || pa >= A.size() || pb >= B.size() || pc >= C.size())
        {
            return false;
        }
        std::memcpy((void*)&output[i], (void*)&A[pa], da*sizeof(float));
        std::memcpy((void*)&output[i+da], (void*)&B[pa], db*sizeof(float));
        std::memcpy((void*)&output[i+da+db], (void*)&C[pc], dc*sizeof(float));
    }

    return true;
}

static float intersectionArea(const Object & a, const Object & b)
{
    cv::Rect a_rect(a.x_offset, a.y_offset, a.width, a.height);
    cv::Rect b_rect(b.x_offset, b.y_offset, b.width, b.height);
    cv::Rect_<float> inter = a_rect & b_rect;
    return inter.area();
}

float limit(float a, float low, float high) { return a < low ? low : (a > high ? high : a); }

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

    output_h_ = std::vector<float>(output_dims_reshape[0] * output_dims_reshape[1] * output_dims_reshape[2]);


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
        checkCudaErrors(cudaMalloc(&mInputTemp1, 1 * 3 * 960 * 960 * sizeof(__half)));
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

void yoloxp::preProcess4Validate(std::vector<cv::Mat> &cv_img)
{
    const double norm_factor_ = 1.0;
    const auto batch_size = cv_img.size();
    const float input_height = static_cast<float>(input_dims[2]);
    const float input_width = static_cast<float>(input_dims[3]);
    std::vector<cv::Mat> dst_images;
    scales_.clear();

    for (const auto & image : cv_img) {
        cv::Mat dst_image;
        const float scale = std::min(input_width / image.cols, input_height / image.rows);

        // std::cout << "scale : " << scale << std::endl;
        // std::cout << "size : " << image.cols * scale << " , " << image.rows * scale << std::endl;

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
        // std::vector<__half> tmp_fp(dim);
        HalfVec tmp_fp(dim);
        convert_float_to_half((float *)imgBuffer, tmp_fp.get(), dim);

        float *a = reinterpret_cast<float *>(imgBuffer);

        std::cout << "img buf : " << a[0] << " " << a[1] << " " << a[2] << std::endl;

        printf("half buf : %f %f %f\n", __half2float(tmp_fp.get()[0]), __half2float(tmp_fp.get()[1]), __half2float(tmp_fp.get()[dim-1]));

        std::cout << "Half size : "  << dim * sizeof(__half) << std::endl;
        // std::cout << "float size : "  << dim * sizeof(float) << std::endl;

        if(mCuDLACtx->getInputCudaBufferPtr(0) != mBindingArray.data())
        {
            std::cout << "dlactx : " << mCuDLACtx->getInputCudaBufferPtr(0) << std::endl;
            std::cout << "mBinding : " << mBindingArray.data() << std::endl;
        }

        checkCudaErrors(cudaMemcpy(mCuDLACtx->getInputCudaBufferPtr(0), (void *)tmp_fp.get(), dim * sizeof(__half), cudaMemcpyHostToDevice));

        // void *input_buffer = mCuDLACtx->getInputCpuBufferPtr(0);

        // if(input_buffer == nullptr)
        // {
        //     std::cout << "null ptr?? " << std::endl;
        // }

        // std::memcpy(input_buffer, tmp_fp.data(), dim*sizeof(__half));

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

    // output_h_.clear();

    if (mBackend == YoloxpBackend::CUDLA_FP16)
    {
        int dim_0 = output_dims_0[0] * output_dims_0[1] * output_dims_0[2];
        std::vector<float> fp_0_float(dim_0);
        copyHalf2Float(fp_0_float, 0);

        int dim_1 = output_dims_1[0] * output_dims_1[1] * output_dims_1[2];
        std::vector<float> fp_1_float(dim_1);
        copyHalf2Float(fp_1_float, 1);
        
        int dim_2 = output_dims_2[0] * output_dims_2[1] * output_dims_2[2];
        std::vector<float> fp_2_float(dim_2);
        copyHalf2Float(fp_2_float, 2);

        if(concat(fp_0_float, fp_1_float, fp_2_float, output_dims_0, output_dims_1, output_dims_2, output_h_))
        {
            std::cout << "concat ok" << std::endl;
        }

        bool empty = true;
        for(int i=0; i<output_h_.size(); ++i)
        {
            if(output_h_[i] != 0.0)
            {
                empty = false;
            }
        }

        if(empty)
        {
            std::cout << "empty output" << std::endl;
        }

        // std::memcopy((void *)&output_h_[0], fp_0_float.data(), dim_0 * sizeof(float));
        // std::memcopy((void *)&output_h_[dim_0], fp_1_float.data(), dim_1 * sizeof(float));
        // std::memcopy((void *)&output_h_[dim_0+dim_1], fp_2_float.data(), dim_2 * sizeof(float));
        
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

std::vector<std::vector<float>> yoloxp::postProcess4Validation()
{
    std::vector<std::vector<float>> res;
    ObjectArray object_array;
    decodeOutputs((float *)output_h_.data(), object_array, scales_[0]);

    for (const auto & object : object_array)
    {
        const auto left = object.x_offset;
        const auto top = object.y_offset;
        const auto right = std::clamp(left + object.width, 0, input_dims[2]);
        const auto bottom = std::clamp(top + object.height, 0, input_dims[3]);
        res.push_back(std::vector<float>{left, top, right, bottom, object.score, object.type});
    }
    return res;
}

void yoloxp::copyHalf2Float(std::vector<float>& out_float, int binding_idx)
{
    int dim_0 = out_float.size();
    std::vector<__half> fp_0(dim_0);
    checkCudaErrors(cudaMemcpy(fp_0.data(), mCuDLACtx->getOutputCudaBufferPtr(binding_idx), dim_0 * sizeof(__half), cudaMemcpyDeviceToHost));
    convert_half_to_float((__half *)fp_0.data(), (float *)out_float.data(), dim_0);
}

void yoloxp::decodeOutputs(
    float * prob, ObjectArray & objects, float scale) const
{
    ObjectArray proposals;
    int img_height = input_dims[2];
    int img_width = input_dims[3];
    const float input_height = static_cast<float>(input_dims[2]);
    const float input_width = static_cast<float>(input_dims[3]);
    std::vector<GridAndStride> grid_strides;
    std::vector<int> output_strides_{8, 16, 32};

    generateGridsAndStride(input_width, input_height, output_strides_, grid_strides);
    generateYoloxProposals(grid_strides, prob, score_threshold_, proposals);

    qsortDescentInplace(proposals);

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
#ifndef __YOLOXP_H__
#define __YOLOXP_H__

#include "NvInfer.h"
#include <NvInferPlugin.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <cstring>

#ifdef USE_DLA_STANDALONE_MODE
#include "cudla_context_standalone.h"
#else
#include "cudla_context_hybrid.h"
#endif

// #include "decode_nms.h"
// #include "matx_reformat.h"

// opencv for preprocessing &  postprocessing
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>

#define EXIT_SUCCESS 0 /* Successful exit status. */
#define EXIT_FAILURE 1 /* Failing exit status.    */

#define checkCudaErrors(call)                                                                                          \
    {                                                                                                                  \
        cudaError_t ret = (call);                                                                                      \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cout << "Cuda failure: " << cudaGetErrorString(ret) << " at line " << __LINE__ << " in file "         \
                      << __FILE__ << " error status: " << ret << std::endl;                                            \
            abort();                                                                                                   \
        }                                                                                                              \
    }

enum YoloxpBackend
{
    TRT_GPU    = 1, // NOT USED
    CUDLA_FP16 = 2,
    CUDLA_INT8 = 3,
};

struct Box
{
    float left, top, right, bottom, confidence;
    float x_offset, y_offset, height, width;
    float class_label;

    Box() = default;

    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label)
    {
    }
};

using BoxArray = std::vector<Box>;

struct Object
{
  int32_t x_offset;
  int32_t y_offset;
  int32_t height;
  int32_t width;
  float score;
  int32_t type;
};

using ObjectArray = std::vector<Object>;

struct GridAndStride
{
  int grid0;
  int grid1;
  int stride;
};

class yoloxp
{
  public:
    //!
    //! \brief init yoloxp class object
    //!
    //! \param engine_path The path of engine/loadable file
    //!
    yoloxp(std::string engine_path, YoloxpBackend backend);

    //!
    //! \brief release yoloxp class object
    //!
    ~yoloxp();

    //!
    //! \brief run tensorRT inference with the data preProcessed
    //!
    int infer();

    // !
    // ! \brief PostProcess, will decode and nms the batch inference result of yoloxp
    // !
    // ! \return return all the nms result of yoloxp
    // !
    std::vector<std::vector<float>> postProcess(float confidence_threshold, float nms_threshold);

    //!
    //! \brief preprocess a list of image for validate mAP on coco dataset! the model must have a [batchsize, 3, 960,
    //! 960] input
    //!
    //! \param cv_img  input images with BGR-UInt8, the size of the vector must smaller than the max batch size of the
    //! model
    //!
    void preProcess4Validate(std::vector<cv::Mat> &cv_img);

    //!
    //! \brief PostProcess for validation on coco dataset, will decode and nms the batch inference result of yoloxp for
    //! mAP test
    //!
    //! \return return all the nms result of yoloxp
    //!
    std::vector<std::vector<float>> postProcess4Validation();

  private:
    int pushImg(void *imgBuffer, int numImg, bool fromCPU = true);

    void copyHalf2Float(std::vector<float>& out_float, int output_idx);

    void decodeOutputs(float * prob, ObjectArray & objects, float scale) const;

    void generateGridsAndStride(const int target_w, const int target_h, const std::vector<int> & strides,
        std::vector<GridAndStride> & grid_strides) const;

    void generateYoloxProposals(std::vector<GridAndStride> grid_strides, float * feat_blob, float prob_threshold,
        ObjectArray & objects) const;
    
    void qsortDescentInplace(ObjectArray & face_objects, int left, int right) const;

    void nmsSortedBboxes(const ObjectArray & face_objects, std::vector<int> & picked, float nms_threshold) const;

    inline void qsortDescentInplace(ObjectArray & objects) const
        {
            if (objects.empty()) {
                return;
            }
            qsortDescentInplace(objects, 0, objects.size() - 1);
        }

  private:
    int mImgPushed;
    int mW;
    int mH;
    float score_threshold_ = 0.3;
    float nms_threshold_ = 0.7;
    int num_class_ = 8;

    std::vector<float> scales_;

    std::vector<int> input_dims{1, 3, 960, 960};
    std::vector<int> output_dims_0{1, 13, 120*120};
    std::vector<int> output_dims_1{1, 13, 60*60};
    std::vector<int> output_dims_2{1, 13, 30*30};

    std::vector<int> output_dims_reshape{1, 18900, 13};

    std::vector<float> input_h_;
    std::vector<float> output_h_;

    cudaStream_t mStream;
    float        ms{0.0f};
    cudaEvent_t  mStartEvent, mEndEvent;

    std::vector<void *> mBindingArray;

    std::string   mEnginePath;
    YoloxpBackend mBackend;
#ifdef USE_DLA_STANDALONE_MODE
    cuDLAContextStandalone *mCuDLACtx;
#else
    cuDLAContextHybrid *mCuDLACtx;
#endif

    // float mInputScale   = 0.00787209;
    // float mOutputScale1 = 0.0546086;
    // float mOutputScale2 = 0.148725;
    // float mOutputScale3 = 0.0546086;
    void * mInputTemp1;
    void * mInputTemp2;

    // chw16 -> chw -> reshape -> transpose operation for yoloxp heads.
    // also support chw -> chw16 for yoloxp inputs
    // ReformatRunner *    mReformatRunner;

    std::vector<void *> src;
    // std::vector<void *> dst{3};

    // For post-processing
    float *                         mAffineMatrix;
    float *                         prior_ptr_dev;
    float *                         parray_host;
    float *                         parray;
    uint64_t                        parray_size;
    std::vector<std::vector<float>> det_results;
    BoxArray                        bas;
};

#endif
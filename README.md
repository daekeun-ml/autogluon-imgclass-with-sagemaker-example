# AutoGluon Image Classification Example
This example shows you how to perform AutoGluon-based Image Classication.

# Prerequisites
- AutoGluon 0.7 
    - Please make sure to install the nightly build 0.7 version(autogluon==0.7.1b20230513), not the 0.7 full version. Object Detection is not possible in the 0.7 full version!
    - Reference: https://github.com/autogluon/autogluon/issues/3082
- Jupyter Notebook (GPU recommended)
    - If you're working in SageMaker, we recommend `ml.g4dn.xlarge` or `ml.g5.xlarge` notebook instance

# Dataset
- This dataset was taken by the author himself with a NVIDIA Jetson nano CSI camera in preparation for AWS IoT Smart Factory Demo and images from the Internet were not used at all. The packaging box images of the dataset are the packaging boxes of Woohwang Cheongsimwon provided with the support of Kwang Dong Pharmaceutical Co.,Ltd.(http://www.ekdp.com), and the camera shooting of the dataset was made with Turck Korea(https://www.turck.kr) inspection equipment sponsored by Turck Korea.

- References
    - [Predictive maintenance use-case: AWS Smart Factory with Turck Korea](https://www.youtube.com/watch?v=R0sMMphzOhw)
    - [Vision Inspection use-case: BIOPLUS-INTERPHEX KOREA (BIX) 2021](https://www.youtube.com/watch?v=iZGa5TRATGQ&t=90s)

## License Summary
This sample code is provided under the MIT-0 license. See the LICENSE file.
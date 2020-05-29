#coding=utf-8
import unittest
import jittor as jt
import numpy as np 
from jittor import Module

ROIALIGN_CPU_HEADER=r"""
#include<cstdio>
#include<cmath>
using namespace std;
"""

ROIALIGN_CPU_SRC=r"""
@alias(images, in0);
@alias(boxes,in1);
@alias(box_ind,in2);
@alias(crops, out0);
        
const float extrapolation_value = @in3(0);
const int batch_size    = images_shape0;
const int depth         = images_shape1;
const int image_height  = images_shape2;
const int image_width   = images_shape3;

const int crop_height = crops_shape2;
const int crop_width = crops_shape3;

#pragma omp parallel for
for (int b = 0; b < boxes_shape0; b++) {
    
    const float y1 = @boxes(b,0);
    const float x1 = @boxes(b,1);
    const float y2 = @boxes(b,2);
    const float x2 = @boxes(b,3);

    const int b_in = @box_ind(b);
            
    if (b_in < 0 || b_in >= batch_size) {
        printf("Error: batch_index %d out of range [0, %d]\n", b_in, batch_size);
        exit(-1);
    }

    const float height_scale = (crop_height > 1)
                    ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                    : 0;
    const float width_scale  = (crop_width > 1) 
                    ? (x2 - x1) * (image_width - 1) / (crop_width - 1)
                    : 0;

    for (int y = 0; y < crop_height; y++){
        const float in_y = (crop_height > 1)
                            ? y1 * (image_height - 1) + y * height_scale
                            : 0.5 * (y1 + y2) * (image_height - 1);

        if (in_y < 0 || in_y > image_height - 1){
                    
            for (int x = 0; x < crop_width; ++x)
                for (int d = 0; d < depth; ++d){
                    // crops(b, y, x, d) = extrapolation_value;
                    @crops(b,d,y,x) = extrapolation_value;
            }

            continue;
        }
                
            
        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        for (int x = 0; x < crop_width; x++){
                
            const float in_x = (crop_width > 1)
                                ? x1 * (image_width - 1) + x * width_scale
                                : 0.5 * (x1 + x2) * (image_width - 1);
            
            if (in_x < 0 || in_x > image_width - 1){
                for (int d = 0; d < depth; ++d){
                        @crops(b,d,y,x) = extrapolation_value;
                }
                continue;
            }
            
            const int left_x_index = floorf(in_x);
            const int right_x_index = ceilf(in_x);
            const float x_lerp = in_x - left_x_index;

            for (int d = 0; d < depth; ++d){   
                    const float top_left = @images(b_in,d,top_y_index,left_x_index);
                    const float top_right = @images(b_in,d,top_y_index,right_x_index);
                    const float bottom_left = @images(b_in,d,bottom_y_index,left_x_index);
                    const float bottom_right = @images(b_in,d,bottom_y_index,right_x_index);
                    
                    const float top = top_left + (top_right - top_left) * x_lerp;
                    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
                        
                    @crops(b,d,y,x) = top + (bottom - top) * y_lerp;
            }
        }   // end for x
    }   // end for y
}   // end for b
"""

ROIALIGN_CPU_GRAD_SRC=[
r"""
@alias(boxes,in1);
@alias(box_ind,in2);
@alias(grads_image, out);
memset(grads_image_p,0,grads_image->size);

const int batch_size    = grads_image_shape0;
const int depth         = grads_image_shape1;
const int image_height  = grads_image_shape2;
const int image_width   = grads_image_shape3;

const int num_boxes     = dout_shape0;
const int crop_height   = dout_shape2;
const int crop_width    = dout_shape3;


for (int b = 0; b < num_boxes; ++b) {

    const float y1 = @boxes(b,0);
    const float x1 = @boxes(b,1);
    const float y2 = @boxes(b,2);
    const float x2 = @boxes(b,3);

    const int b_in = @box_ind(b);

    if (b_in < 0 || b_in >= batch_size) {
        printf("Error: batch_index %d out of range [0, %d]\n", b_in, batch_size);
        exit(-1);
    }

    const float height_scale = (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1): 0;
    const float width_scale = (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1): 0;

    for (int y = 0; y < crop_height; ++y){
        const float in_y = (crop_height > 1)
                                   ? y1 * (image_height - 1) + y * height_scale
                                   : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1){
                continue;
        }
        
        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        for (int x = 0; x < crop_width; ++x){
            
            const float in_x = (crop_width > 1)
                                       ? x1 * (image_width - 1) + x * width_scale
                                       : 0.5 * (x1 + x2) * (image_width - 1);
            if (in_x < 0 || in_x > image_width - 1){
                    continue;
            }
            const int left_x_index = floorf(in_x);
            const int right_x_index = ceilf(in_x);
            const float x_lerp = in_x - left_x_index;

            for (int d = 0; d < depth; ++d){
                    const float grad_val = @dout(b,d,y,x);
                    const float dtop = (1 - y_lerp) * grad_val;
                    @grads_image(b_in,d,top_y_index,left_x_index) += (1 - x_lerp) * dtop;
                    @grads_image(b_in,d,top_y_index,right_x_index) += x_lerp * dtop;

                    const float dbottom = y_lerp * grad_val;
                    @grads_image(b_in,d,bottom_y_index,left_x_index) += (1 - x_lerp) * dbottom;
                    @grads_image(b_in,d,bottom_y_index,right_x_index) += x_lerp * dbottom;
            }   // end d
        }   // end x
    }   // end y
}   // end b
""",
"","",""]


ROIALIGN_CUDA_HEADER=r"""
#include <cmath>
#include <cstdio>
using namespace std;
"""



ROIALIGN_CUDA_SRC=r"""

__global__ static void CropAndResizeKernel(@ARGS_DEF){
    @PRECALC
    @alias(images, in0);
    @alias(boxes,in1);
    @alias(box_ind,in2);
    @alias(crops, out0);
    

    const int batch_size    = images_shape0;
    const int depth         = images_shape1;
    const int image_height  = images_shape2;
    const int image_width   = images_shape3;

    const int num_boxes     = boxes_shape0;
    const int crop_height = crops_shape2;
    const int crop_width = crops_shape3;
    const int total_count = num_boxes * crop_height * crop_width * depth;

    const float extrapolation_value = @in3(0);

    for (int out_idx = blockIdx.x * blockDim.x + threadIdx.x; out_idx < total_count; out_idx += blockDim.x * gridDim.x){
        // NHWC: out_idx = d + depth * (w + crop_width * (h + crop_height * b))
        // NCHW: out_idx = w + crop_width * (h + crop_height * (d + depth * b))
        int idx = out_idx;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int d = idx % depth;
        const int b = idx / depth;

        const float y1 = @boxes(b,0);
        const float x1 = @boxes(b,1);
        const float y2 = @boxes(b,2);
        const float x2 = @boxes(b,3);

        const int b_in = @box_ind(b);
        if (b_in < 0 || b_in >= batch_size){
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

        const float in_y = (crop_height > 1)
                                ? y1 * (image_height - 1) + y * height_scale
                                : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1){
            crops_p[out_idx] = extrapolation_value;
            continue;
        }

        const float in_x = (crop_width > 1)
                                ? x1 * (image_width - 1) + x * width_scale
                                : 0.5 * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1)
        {
            crops_p[out_idx] = extrapolation_value;
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        const float *pimage = images_p + (b_in * depth + d) * image_height * image_width;
        const float top_left = pimage[top_y_index * image_width + left_x_index];
        const float top_right = pimage[top_y_index * image_width + right_x_index];
        const float bottom_left = pimage[bottom_y_index * image_width + left_x_index];
        const float bottom_right = pimage[bottom_y_index * image_width + right_x_index];

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        crops_p[out_idx] = top + (bottom - top) * y_lerp;
    }
}

const int num_boxes     = out0_shape0;
const int depth      = out0_shape1;
const int crop_height = out0_shape2;
const int crop_width = out0_shape3;

memset(out0_p,0,out0->size);

const int total_count = num_boxes * crop_height * crop_width * depth;
const int thread_per_block = 1024;
const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
CropAndResizeKernel<<<block_count, thread_per_block>>>(@ARGS);

"""

ROIALIGN_CUDA_GRAD_SRC=[r"""
__global__ static void CropAndResizeBackpropImageKernel(@ARGS_DEF){
    @PRECALC
    const int num_boxes     = dout_shape0;
    const int depth = dout_shape1;
    const int crop_height = dout_shape2;
    const int crop_width = dout_shape3;
    const int total_count = num_boxes * crop_height * crop_width * depth;

    @alias(boxes,in1);
    @alias(box_ind,in2);
    @alias(grads_image, out);

    const int batch_size    = grads_image_shape0;
    const int image_height  = grads_image_shape2;
    const int image_width   = grads_image_shape3;

    for (int out_idx = blockIdx.x * blockDim.x + threadIdx.x; out_idx < total_count; out_idx += blockDim.x * gridDim.x){
        int idx = out_idx;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int d = idx % depth;
        const int b = idx / depth;

        const float y1 = @boxes(b,0);
        const float x1 = @boxes(b,1);
        const float y2 = @boxes(b,2);
        const float x2 = @boxes(b,3);

        const int b_in = @box_ind(b);
        if (b_in < 0 || b_in >= batch_size)
        {
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

        const float in_y = (crop_height > 1)
                                ? y1 * (image_height - 1) + y * height_scale
                                : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1)
        {
            continue;
        }

        const float in_x = (crop_width > 1)
                                ? x1 * (image_width - 1) + x * width_scale
                                : 0.5 * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1)
        {
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        float *pimage = grads_image_p + (b_in * depth + d) * image_height * image_width;
        const float dtop = (1 - y_lerp) * dout_p[out_idx];
        atomicAdd(
            pimage + top_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dtop
        );
        atomicAdd(
            pimage + top_y_index * image_width + right_x_index, 
            x_lerp * dtop
        );

        const float dbottom = y_lerp * dout_p[out_idx];
        atomicAdd(
            pimage + bottom_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dbottom
        );
        atomicAdd(
            pimage + bottom_y_index * image_width + right_x_index, 
            x_lerp * dbottom
        );

    }
}


const int num_boxes     = dout_shape0;
const int depth      = dout_shape1;
const int crop_height = dout_shape2;
const int crop_width = dout_shape3;

memset(out0_p,0,out0->size);

const int total_count = num_boxes * crop_height * crop_width * depth;
const int thread_per_block = 1024;
const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
CropAndResizeBackpropImageKernel<<<block_count, thread_per_block>>>(@ARGS);

""","","",""]


class RoIAlign(Module):
    '''This is ported from https://github.com/longcw/RoIAlign.pytorch '''

    def __init__(self,crop_height, crop_width, extrapolation_value=0, transform_fpcoor=True):
        super (RoIAlign, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value
        self.transform_fpcoor = transform_fpcoor

    def execute(self,featuremap, boxes, box_ind):
        """
        RoIAlign based on crop_and_resize.
        See more details on https://github.com/longcw/RoIAlign.pytorch
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """
        x1, y1, x2, y2 = [boxes.reindex([boxes.shape[0],1], ["i0", str(i)]) for i in range(4)]
        image_height, image_width = featuremap.shape[2:4]

        if self.transform_fpcoor:
            spacing_w = (x2 - x1) / float(self.crop_width)
            spacing_h = (y2 - y1) / float(self.crop_height)

            nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
            nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
            nh = spacing_h * float(self.crop_height - 1) / float(image_height - 1)

            boxes = jt.contrib.concat((ny0, nx0, ny0 + nh, nx0 + nw), 1)
        else:
            x1 = x1 / float(image_width - 1)
            x2 = x2 / float(image_width - 1)
            y1 = y1 / float(image_height - 1)
            y2 = y2 / float(image_height - 1)
            boxes = jt.contrib.concat((y1, x1, y2, x2), 1)

        num_boxes = boxes.shape[0]
        depth = featuremap.shape[1]

        output_shapes = (num_boxes, depth, self.crop_height, self.crop_width)
        output_types = featuremap.dtype
        extrapolation_value = jt.array([self.extrapolation_value])
        inputs = [featuremap,boxes,box_ind,extrapolation_value]
        cpu_header = ROIALIGN_CPU_HEADER
        cpu_src =ROIALIGN_CPU_SRC
        cpu_grad_src = ROIALIGN_CPU_GRAD_SRC

        cuda_header = ROIALIGN_CUDA_HEADER
        cuda_src= ROIALIGN_CUDA_SRC
        cuda_grad_src= ROIALIGN_CUDA_GRAD_SRC

        output = jt.code(output_shapes,output_types,inputs,cpu_header = cpu_header,
            cpu_src=cpu_src,cpu_grad_src=cpu_grad_src,cuda_header=cuda_header,cuda_src=cuda_src,cuda_grad_src=cuda_grad_src)

        return output


def test_roialign(images,boxes,box_index,crop_height = 4,crop_width = 4):
    # roi_align is from https://github.com/longcw/RoIAlign.pytorch
    # install it and test
    skip_test = False
    try:
        import roi_align
        import torch
    except:
        skip_test = True

    if skip_test:
       print("Please install RoIAlign.pytorch")
       return


    jt_image = jt.array(images)
    torch_image = torch.from_numpy(images).float()
    torch_image.requires_grad=True
    assert np.allclose(jt_image.data,torch_image.detach().numpy())

    jt_boxes  = jt.array(boxes)
    torch_boxes = torch.from_numpy(boxes).float()
    assert np.allclose(jt_boxes.data,torch_boxes.detach().numpy())

    jt_box_index = jt.array(box_index)
    torch_box_index = torch.from_numpy(box_index).int()
    assert np.allclose(jt_box_index.data,torch_box_index.detach().numpy())

    # build roialign module
    torch_roi_align = roi_align.RoIAlign(crop_height, crop_width)
    jt_roi_align = RoIAlign(crop_height,crop_width)

    # make crops
    torch_crops = torch_roi_align(torch_image, torch_boxes, torch_box_index)
    jt_crops = jt_roi_align(jt_image,jt_boxes,jt_box_index)

    max_error = np.max(np.abs(jt_crops.data-torch_crops.detach().numpy()))
    assert np.allclose(jt_crops.data,torch_crops.detach().numpy()) or max_error<1e-5

    # make grads
    grad = torch.ones(torch_crops.shape)
    torch_crops.backward(gradient=grad)
    jt_image_grad = jt.grad(jt_crops,jt_image)

    max_error = np.max(np.abs(jt_image_grad.data-torch_image.grad.detach().numpy()))
    assert np.allclose(jt_image_grad.data,torch_image.grad.detach().numpy()) or max_error<1e-5
    


class TestRoIAlign(unittest.TestCase):
    def test(self):
        jt.dirty_fix_pytorch_runtime_error()
        test_times = 1000
        for i in range(test_times):
            images = np.random.randn(2,1,7,7).astype(np.float32)
            boxes = np.array([[1, 0, 5, 4],
                     [0.5, 3.5, 4, 7]])
            box_index = np.array([1,0],np.int)

            jt.flags.use_cuda = 0
            test_roialign(images,boxes,box_index)

            jt.flags.use_cuda = 1
            test_roialign(images,boxes,box_index)

if __name__ == '__main__':
    unittest.main()
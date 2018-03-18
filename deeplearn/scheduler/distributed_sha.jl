
using Cxx
using PyCall

# cvlibdir = "/usr/local/lib/"
# cvheaderdir = "/usr/local/include/"

version = "3.4.1"

libprefix = "libopencv_"

libNames = [
              "shape",
              "stitching",
              "objdetect",
              "superres",
              "videostab",
              "calib3d",
              "features2d",
              "highgui",
              "videoio",
              "imgcodecs",
              "video",
              "photo",
              "ml",
              "imgproc",
              #"flann",   // TODO: resolve typeid error due to rtti flag
              #"viz"
            ]

function getFullLibNames()
    return map(x -> "$(libprefix)$(x).$(version).dylib", libNames)
end

LIB_PATH = "/usr/local/Cellar/opencv/3.4.1_2/lib/"
INCLUDE_PATH = "/usr/local/Cellar/opencv/3.4.1_2/include/"

addHeaderDir(INCLUDE_PATH, kind = C_System)

for i in getFullLibNames()
    if is_linux()
         i = swapext(i[1:end-6], ".so")
    end
    # Must link symbols accross libraries with RTLD_GLOBAL
    Libdl.dlopen(joinpath(LIB_PATH,i), Libdl.RTLD_GLOBAL)
end

addHeaderDir(joinpath(INCLUDE_PATH,"opencv2"), kind = C_System )
addHeaderDir(joinpath(INCLUDE_PATH,"opencv2/core"), kind = C_System )
#cxxinclude(joinpath(INCLUDE_PATH,"opencv2/opencv.hpp"))

#cxxinclude(joinpath(cvheaderdir,"opencv2/core/opengl.hpp"))            # enable OpenGL
#cxxinclude(joinpath(cvheaderdir,"opencv2/core/ocl.hpp"))               # enable OpenCL
#cxxinclude(joinpath(cvheaderdir,"opencv2/video/background_segm.hpp"))  # enable bg/fg segmentation
#cxxinclude(joinpath(cvheaderdir,"opencv2/video/tracking.hpp"))         # enable tracking
#cxxinclude(joinpath(cvheaderdir,"opencv2/shape.hpp"))
#cxxinclude(joinpath(cvheaderdir,"opencv2/stitching.hpp"))
#cxxinclude(joinpath(cvheaderdir,"opencv2/superres.hpp"))
#cxxinclude(joinpath(cvheaderdir,"opencv2/videostab.hpp"))

cxx"""
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
"""

v = "/Volumes/Media/south-park/South.Park.S19E06.INTERNAL.720p.HDTV.x264-KILLERS[eztv].mp4"

mutable struct VideoStream
    cxx_handle::Any
    width::UInt64
    height::UInt64
    fps::Float64
    duration::Float64
    num_frames::UInt64
end


function open_video(file::AbstractString)
    video = icxx"""
    cv::VideoCapture video_file($file);

    return video_file;
    """
    return VideoStream(
        video,
        UInt64(@cxx video->get(@cxx cv::CAP_PROP_FRAME_WIDTH)),
        UInt64(@cxx video->get(@cxx cv::CAP_PROP_FRAME_HEIGHT)),
        Float64(@cxx video->get(@cxx cv::CAP_PROP_FPS)),
        0,
        (@cxx video->get(@cxx cv::CAP_PROP_FRAME_COUNT))
    )
end

struct VideoFrame
    cxx_mat::Any
    width::UInt64
    height::UInt64
    data::Array{UInt8}
end

#/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia distributed_sha.jl

function read_frame(stream::VideoStream, pos::UInt64)
    frame = icxx"""
    cv::Mat frame;
    $(stream.cxx_handle).set(cv::CAP_PROP_POS_MSEC, $pos);
    $(stream.cxx_handle) >> frame;
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB );
    return frame;
    """
    data_length = (@cxx frame->dataend) - (@cxx frame->datastart)
    data = unsafe_wrap(Array, (@cxx frame->data), data_length, true)
    return VideoFrame(
        frame,
        (@cxx frame->cols),
        (@cxx frame->rows),
        data
    )
end

@pyimport PIL.Image as Image

foo = open_video(v)
fr = read_frame(foo, UInt64(100000))
size = fr.width*fr.height*3
mode = py"'RGB'"
@show Image.Image
im = @pycall Image.new("RGB", (fr.width, fr.height)) :: PyObject #.frombytes(mode, (fr.width, fr.height), fr.data) #reinterpret(Array{UInt8, 1}, fr.data))
@show im
#im[:mode] = "RGB"
#im[:size] = (fr.width, fr.height)
im[:frombytes](pybytes(fr.data))
#im = Image.fromarray(PyObject(fr.data), mode) #, (fr.width, fr.height), fr.data)
im[:save]("foobar.jpg", "JPEG")
# @show fr.data[1:1000]
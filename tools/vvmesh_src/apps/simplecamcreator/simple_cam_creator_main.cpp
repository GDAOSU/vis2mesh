// Copyright 2019 Shaun Song <sxsong1207@qq.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <pcl/console/print.h>

#include <opencv2/core.hpp>

#include "simple_viewer.h"
#include "param.h"

#if WIN32
#include <windows.h>
#else
#include <X11/Xlib.h>
#endif

void getScreenResolution(int &width, int &height) {
#if WIN32
  width = (int)GetSystemMetrics(SM_CXSCREEN);
  height = (int)GetSystemMetrics(SM_CYSCREEN);
#else
  Display *disp = XOpenDisplay(NULL);
  Screen *scrn = DefaultScreenOfDisplay(disp);
  width = scrn->width;
  height = scrn->height;
#endif
}

int main(int argc, char **argv) {
  ParamsSet Params;
  pcl::console::setVerbosityLevel(pcl::console::L_INFO);
  cv::String keys =
      "{help h usage ?  |     | print this message}"

      "{input_pts ip    |     | input cloud}"
      "{input_tex tex   |     | input trimesh}"
      "{input_poly poly |     | input polymesh}"
      "{input_list il   |     | input json}"
      "{output_list ol  |     | output path}"
      //-------- Camera Model-------
      "{focal         | 800 | focal length}"
      "{width         | 2000| camera width}"
      "{height        | 1500| camera height}"
      //-------- Map Projection and filter
      "{sample samp    |999999| down sample cloud to display}"
      "{resolution  res |  10 | octree resolution}"
      "{winwidth        |99999| max win width}"
      "{winheight       |99999| max win height}"

      //------- Camera Generator -----
      "{camgenheight       | 50  | camera pos generator default height }"
      "{camgenoverlap      | 0.0 | camera pos generator default overlap rate }";

  cv::CommandLineParser clp(argc, argv, keys);
  clp.about("Virtual Pose Viewer for MVS, created by Shaun Song");
  if (clp.has("help")) {
    clp.printMessage();
    return 0;
  }
  Params.Input_pts = clp.get<std::string>("input_pts");
  Params.Input_texmesh = clp.get<std::string>("input_tex");
  Params.Input_polymesh = clp.get<std::string>("input_poly");
  Params.Input_list = clp.get<std::string>("input_list");
  Params.Output_list = clp.get<std::string>("output_list");

  if (Params.Input_list.empty()) {
    Params.Input_list = Params.Output_list;
  }

  Params.focalLength = clp.get<double>("focal");
  Params.ImageWidth = clp.get<int>("width");
  Params.ImageHeight = clp.get<int>("height");
  Params.WHRatio = (double)Params.ImageWidth / (double)Params.ImageHeight;
  Params.OctreeResolution = clp.get<float>("resolution");
  Params.SampleToDisplay = clp.get<int>("sample");

  {  // make k
    double prin_x = Params.ImageWidth / 2;
    double prin_y = Params.ImageHeight / 2;
    Params.K << Params.focalLength, 0, prin_x, 0, Params.focalLength, prin_y, 0,
        0, 1;
    Params.InvK = Params.K.inverse();
    Params.fovy = 2 * atan(Params.ImageHeight / (2. * Params.focalLength));
  }
  Params.nearClipping = 0.1;  // clp.get<float>("znear");
  Params.farClipping = 1000;  // clp.get<float>("zfar");

  {
    int widthlimit = clp.get<int>("winwidth");
    int heightlimit = clp.get<int>("winheight");
    int scr_width, scr_height;
    getScreenResolution(scr_width, scr_height);
    // determine good
    double max_width = std::min(scr_width - 20, widthlimit),
           max_height = std::min(scr_height - 100, heightlimit);

    int window_width, window_height;
    double whratio = Params.WHRatio;
    double height_from_max_width = max_width / whratio;
    double width_from_max_height = max_height * whratio;
    if (height_from_max_width > max_height) {
      window_height = max_height;
      window_width = width_from_max_height;
    } else {
      window_height = height_from_max_width;
      window_width = max_width;
    }
    Params.WinWidth = window_width;
    Params.WinHeight = window_height;
    Params.ScaleRealDisplay = (double)Params.ImageWidth / (double)window_width;
  }
  Params.CamGen_height = clp.get<float>("camgenheight");
  Params.CamGen_overlap = clp.get<float>("camgenoverlap");

  Params.print();
  SimpleCamViewer v(Params);
  v.run();
  return 0;
}

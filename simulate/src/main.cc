// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// !!! hack code: make glfw_adapter.window_ public
#define private public
#include "glfw_adapter.h"
#undef private

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include <atomic>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <mujoco/mujoco.h>
#include "simulate.h"
#include "array_safety.h"
#include "unitree_sdk2_bridge.h"
#include "param.h"
#include "cmg_viz_shm.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"
#define NUM_MOTOR_IDL_GO 20

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

class ElasticBand
{
public:
  ElasticBand(){};
  void Advance(std::vector<double> x, std::vector<double> dx)
  {
    std::vector<double> delta_x = {0.0, 0.0, 0.0};
    delta_x[0] = point_[0] - x[0];
    delta_x[1] = point_[1] - x[1];
    delta_x[2] = point_[2] - x[2];
    double distance = sqrt(delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2]);

    std::vector<double> direction = {0.0, 0.0, 0.0};
    direction[0] = delta_x[0] / distance;
    direction[1] = delta_x[1] / distance;
    direction[2] = delta_x[2] / distance;

    double v = dx[0] * direction[0] + dx[1] * direction[1] + dx[2] * direction[2];

    f_[0] = (stiffness_ * (distance - length_) - damping_ * v) * direction[0];
    f_[1] = (stiffness_ * (distance - length_) - damping_ * v) * direction[1];
    f_[2] = (stiffness_ * (distance - length_) - damping_ * v) * direction[2];
  }


  double stiffness_ = 200;
  double damping_ = 100;
  std::vector<double> point_ = {0, 0, 3};
  double length_ = 0.0;
  bool enable_ = true;
  std::vector<double> f_ = {0, 0, 0};
};
inline ElasticBand elastic_band;


namespace
{
  namespace mj = ::mujoco;
  namespace mju = ::mujoco::sample_util;

  // constants
  const double syncMisalign = 0.1;       // maximum mis-alignment before re-sync (simulation seconds)
  const double simRefreshFraction = 0.7; // fraction of refresh available for simulation
  const int kErrorLength = 1024;         // load error string length

  // model and data
  mjModel *m = nullptr;
  mjData *d = nullptr;

  // camera and viz options (file scope so PhysicsThread can configure them after model load)
  mjvCamera cam;
  mjvOption opt;

  // control noise variables
  mjtNum *ctrlnoise = nullptr;

  using Seconds = std::chrono::duration<double>;

  //---------------------------------------- plugin handling -----------------------------------------

  // return the path to the directory containing the current executable
  // used to determine the location of auto-loaded plugin libraries
  std::string getExecutableDir()
  {
#if defined(_WIN32) || defined(__CYGWIN__)
    constexpr char kPathSep = '\\';
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      DWORD buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
        if (written < buf_size)
        {
          success = true;
        }
        else if (written == buf_size)
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
        else
        {
          std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
          return "";
        }
      }
      return realpath.get();
    }();
#else
    constexpr char kPathSep = '/';
#if defined(__APPLE__)
    std::unique_ptr<char[]> buf(nullptr);
    {
      std::uint32_t buf_size = 0;
      _NSGetExecutablePath(nullptr, &buf_size);
      buf.reset(new char[buf_size]);
      if (!buf)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      if (_NSGetExecutablePath(buf.get(), &buf_size))
      {
        std::cerr << "unexpected error from _NSGetExecutablePath\n";
      }
    }
    const char *path = buf.get();
#else
    const char *path = "/proc/self/exe";
#endif
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      std::uint32_t buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        std::size_t written = readlink(path, realpath.get(), buf_size);
        if (written < buf_size)
        {
          realpath.get()[written] = '\0';
          success = true;
        }
        else if (written == -1)
        {
          if (errno == EINVAL)
          {
            // path is already not a symlink, just use it
            return path;
          }

          std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
          return "";
        }
        else
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
      }
      return realpath.get();
    }();
#endif

    if (realpath.empty())
    {
      return "";
    }

    for (std::size_t i = realpath.size() - 1; i > 0; --i)
    {
      if (realpath.c_str()[i] == kPathSep)
      {
        return realpath.substr(0, i);
      }
    }

    // don't scan through the entire file system's root
    return "";
  }

  // scan for libraries in the plugin directory to load additional plugins
  void scanPluginLibraries()
  {
    // check and print plugins that are linked directly into the executable
    int nplugin = mjp_pluginCount();
    if (nplugin)
    {
      std::printf("Built-in plugins:\n");
      for (int i = 0; i < nplugin; ++i)
      {
        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
      }
    }

    // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
    const std::string sep = "\\";
#else
    const std::string sep = "/";
#endif

    // try to open the ${EXECDIR}/plugin directory
    // ${EXECDIR} is the directory containing the simulate binary itself
    const std::string executable_dir = getExecutableDir();
    if (executable_dir.empty())
    {
      return;
    }

    const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
    mj_loadAllPluginLibraries(
        plugin_dir.c_str(), +[](const char *filename, int first, int count)
                            {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        } });
  }

  //------------------------------------------- simulation -------------------------------------------

  mjModel *LoadModel(const char *file, mj::Simulate &sim)
  {
    // this copy is needed so that the mju::strlen call below compiles
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    // make sure filename is not empty
    if (!filename[0])
    {
      return nullptr;
    }

    // load and compile
    char loadError[kErrorLength] = "";
    mjModel *mnew = 0;
    if (mju::strlen_arr(filename) > 4 &&
        !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                      mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
      mnew = mj_loadModel(filename, nullptr);
      if (!mnew)
      {
        mju::strcpy_arr(loadError, "could not load binary model");
      }
    }
    else
    {
      mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
      // remove trailing newline character from loadError
      if (loadError[0])
      {
        int error_length = mju::strlen_arr(loadError);
        if (loadError[error_length - 1] == '\n')
        {
          loadError[error_length - 1] = '\0';
        }
      }
    }

    mju::strcpy_arr(sim.load_error, loadError);

    if (!mnew)
    {
      std::printf("%s\n", loadError);
      return nullptr;
    }

    // compiler warning: print and pause
    if (loadError[0])
    {
      // mj_forward() below will print the warning message
      std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
      sim.run = 0;
    }

    return mnew;
  }

  // simulate in background thread (while rendering in main thread)
  void PhysicsLoop(mj::Simulate &sim)
  {
    // cpu-sim syncronization point
    std::chrono::time_point<mj::Simulate::Clock> syncCPU;
    mjtNum syncSim = 0;

    // ChannelFactory::Instance()->Init(0);
    // UnitreeDds ud(d);

    // run until asked to exit
    while (!sim.exitrequest.load())
    {
      if (sim.droploadrequest.load())
      {
        sim.LoadMessage(sim.dropfilename);
        mjModel *mnew = LoadModel(sim.dropfilename, sim);
        sim.droploadrequest.store(false);

        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.dropfilename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = (mjtNum *)malloc(sizeof(mjtNum) * m->nu);
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      if (sim.uiloadrequest.load())
      {
        sim.uiloadrequest.fetch_sub(1);
        sim.LoadMessage(sim.filename);
        mjModel *mnew = LoadModel(sim.filename, sim);
        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.filename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      // sleep for 1 ms or yield, to let main thread run
      //  yield results in busy wait - which has better timing but kills battery life
      if (sim.run && sim.busywait)
      {
        std::this_thread::yield();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      {
        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        // run only if model is present
        if (m)
        {
          // running
          if (sim.run)
          {
            bool stepped = false;

            // record cpu time at start of iteration
            const auto startCPU = mj::Simulate::Clock::now();

            // elapsed CPU and simulation time since last sync
            const auto elapsedCPU = startCPU - syncCPU;
            double elapsedSim = d->time - syncSim;

            // inject noise
            if (sim.ctrl_noise_std)
            {
              // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
              mjtNum rate = mju_exp(-m->opt.timestep / mju_max(sim.ctrl_noise_rate, mjMINVAL));
              mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

              for (int i = 0; i < m->nu; i++)
              {
                // update noise
                ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

                // apply noise
                d->ctrl[i] = ctrlnoise[i];
              }
            }

            // requested slow-down factor
            double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

            // misalignment condition: distance from target sim time is bigger than syncmisalign
            bool misaligned =
                mju_abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

            // out-of-sync (for any reason): reset sync times, step
            if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                misaligned || sim.speed_changed)
            {
              // re-sync
              syncCPU = startCPU;
              syncSim = d->time;
              sim.speed_changed = false;

              // run single step, let next iteration deal with timing
              mj_step(m, d);
              stepped = true;
            }

            // in-sync: step until ahead of cpu
            else
            {
              bool measured = false;
              mjtNum prevSim = d->time;

              double refreshTime = simRefreshFraction / sim.refresh_rate;

              // step while sim lags behind cpu and within refreshTime
              while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                     mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
              {
                // measure slowdown before first step
                if (!measured && elapsedSim)
                {
                  sim.measured_slowdown =
                      std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                  measured = true;
                }

                // elastic band on base link
                if (param::config.enable_elastic_band == 1)
                {
                  if (elastic_band.enable_)
                  {
                    std::vector<double> x = {d->qpos[0], d->qpos[1], d->qpos[2]};
                    std::vector<double> dx = {d->qvel[0], d->qvel[1], d->qvel[2]};

                    elastic_band.Advance(x, dx);

                    d->xfrc_applied[param::config.band_attached_link] = elastic_band.f_[0];
                    d->xfrc_applied[param::config.band_attached_link + 1] = elastic_band.f_[1];
                    d->xfrc_applied[param::config.band_attached_link + 2] = elastic_band.f_[2];
                  }
                }

                // call mj_step
                mj_step(m, d);
                stepped = true;

                // break if reset
                if (d->time < prevSim)
                {
                  break;
                }
              }
            }

            // save current state to history buffer
            if (stepped)
            {
              sim.AddToHistory();
            }
          }

          // paused
          else
          {
            // run mj_forward, to update rendering and joint sliders
            mj_forward(m, d);
            sim.speed_changed = true;
          }
        }
      } // release std::lock_guard<std::mutex>
    }
  }
} // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename)
{
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr)
  {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m)
      d = mj_makeData(m);
    if (d)
    {
      sim->Load(m, d, filename);
      mj_forward(m, d);

      // Lock camera to robot torso/pelvis so view follows the body on open.
      int track_bid = mj_name2id(m, mjOBJ_BODY, "torso_link");
      if (track_bid < 0) track_bid = mj_name2id(m, mjOBJ_BODY, "pelvis");
      if (track_bid < 0) track_bid = (m->nbody > 1) ? 1 : 0;
      cam.type = mjCAMERA_TRACKING;
      cam.trackbodyid = track_bid;
      cam.distance = 3.0;
      cam.lookat[2] = 0.6;
      std::printf("[Camera] tracking body id=%d (%s), distance=%.1f\n",
                  track_bid, mj_id2name(m, mjOBJ_BODY, track_bid), cam.distance);

      // allocate ctrlnoise
      free(ctrlnoise);
      ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
      mju_zero(ctrlnoise, m->nu);
    }
    else
    {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);

  exit(0);
}

void *UnitreeSdk2BridgeThread(void *arg)
{
  // Wait for mujoco data
  while (true)
  {
    if (d)
    {
      std::cout << "Mujoco data is prepared" << std::endl;
      break;
    }
    usleep(500000);
  }

  unitree::robot::ChannelFactory::Instance()->Init(param::config.domain_id, param::config.interface);


  int body_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
  if (body_id < 0) {
    body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
  }
  param::config.band_attached_link = 6 * body_id;
  
  std::unique_ptr<UnitreeSDK2BridgeBase> interface = nullptr;
  if (m->nu > NUM_MOTOR_IDL_GO) {
    interface = std::make_unique<G1Bridge>(m, d);
  } else {
    interface = std::make_unique<Go2Bridge>(m, d);
  }
  interface->start();
  
  while (true)
  {
    sleep(1);
  }
}

//---------------------------------------- CMG visualisation ---------------------------------------

std::atomic<bool> cmg_viz_enabled{false};

// // Ring-buffer sample used by the viz thread
// struct CmgSample {
//   float qref[CMG_VIZ_NUM_JOINTS];
//   float actual[CMG_VIZ_NUM_JOINTS];
//   float command[CMG_VIZ_CMD_DIM];
// };

// static void BuildLegFigure(mjvFigure& fig, const std::deque<CmgSample>& ring,
//                            const int* indices, int n_idx, const char* title,
//                            const char* joint_names[][2]) {
//   mjv_defaultFigure(&fig);
//   fig.flg_extend = 1;
//   fig.flg_legend = 1;
//   snprintf(fig.title, sizeof(fig.title), "%s", title);

//   // Line layout: for each joint i we have line 2*i = qref, line 2*i+1 = actual
//   int n_lines = 2 * n_idx;
//   fig.linepnt[0] = 0;  // will be set below
//   for (int i = 0; i < n_idx; i++) {
//     int l_ref = 2 * i;
//     int l_act = 2 * i + 1;

//     // Line names
//     snprintf(fig.linename[l_ref], sizeof(fig.linename[0]), "%s ref", joint_names[i][0]);
//     snprintf(fig.linename[l_act], sizeof(fig.linename[0]), "%s act", joint_names[i][0]);

//     // Colors — ref: bright, actual: dim version of same hue
//     // Use distinct hues per joint
//     static const float hues[][3] = {
//       {1.0f, 0.2f, 0.2f},  // red
//       {0.2f, 0.8f, 0.2f},  // green
//       {0.3f, 0.3f, 1.0f},  // blue
//       {1.0f, 0.8f, 0.0f},  // yellow
//       {1.0f, 0.4f, 0.8f},  // pink
//       {0.0f, 0.8f, 0.8f},  // cyan
//     };
//     for (int c = 0; c < 3; c++) {
//       fig.linergb[l_ref][c] = hues[i % 6][c];
//       fig.linergb[l_act][c] = hues[i % 6][c] * 0.45f;
//     }

//     // Fill data from ring buffer
//     int n_pts = static_cast<int>(ring.size());
//     if (n_pts > 1000) n_pts = 1000;  // mjvFigure max is 1001
//     fig.linepnt[l_ref] = n_pts;
//     fig.linepnt[l_act] = n_pts;
//     int jdx = indices[i];
//     for (int t = 0; t < n_pts; t++) {
//       const auto& s = ring[ring.size() - n_pts + t];
//       float x = static_cast<float>(t);
//       fig.linedata[l_ref][2 * t]     = x;
//       fig.linedata[l_ref][2 * t + 1] = s.qref[jdx];
//       fig.linedata[l_act][2 * t]     = x;
//       fig.linedata[l_act][2 * t + 1] = s.actual[jdx];
//     }
//   }
//   // Zero out remaining lines
//   for (int l = n_lines; l < mjMAXLINE; l++) {
//     fig.linepnt[l] = 0;
//   }
// }

// static void BuildCommandFigure(mjvFigure& fig, const std::deque<CmgSample>& ring) {
//   mjv_defaultFigure(&fig);
//   fig.flg_extend = 1;
//   fig.flg_legend = 1;
//   snprintf(fig.title, sizeof(fig.title), "Velocity Commands");

//   static const char* names[] = {"vx", "vy", "yaw_rate"};
//   static const float colors[][3] = {
//     {1.0f, 0.3f, 0.3f},
//     {0.3f, 1.0f, 0.3f},
//     {0.3f, 0.3f, 1.0f},
//   };

//   int n_pts = static_cast<int>(ring.size());
//   if (n_pts > 1000) n_pts = 1000;

//   for (int i = 0; i < 3; i++) {
//     snprintf(fig.linename[i], sizeof(fig.linename[0]), "%s", names[i]);
//     for (int c = 0; c < 3; c++) fig.linergb[i][c] = colors[i][c];
//     fig.linepnt[i] = n_pts;
//     for (int t = 0; t < n_pts; t++) {
//       const auto& s = ring[ring.size() - n_pts + t];
//       fig.linedata[i][2 * t]     = static_cast<float>(t);
//       fig.linedata[i][2 * t + 1] = s.command[i];
//     }
//   }
//   for (int l = 3; l < mjMAXLINE; l++) {
//     fig.linepnt[l] = 0;
//   }
// }

// void CmgVizThread(mj::Simulate* sim) {
//   CMGVizReader reader;
//   std::deque<CmgSample> ring;
//   const int MAX_RING = 200;

//   // USD joint indices for left / right leg
//   const int left_idx[]  = {0, 3, 6, 9, 13, 17};
//   const int right_idx[] = {1, 4, 7, 10, 14, 18};
//   const int N_LEG = 6;

//   const char* left_names[][2]  = {{"L hip_p",""}, {"L hip_r",""}, {"L hip_y",""},
//                                    {"L knee",""}, {"L ank_p",""}, {"L ank_r",""}};
//   const char* right_names[][2] = {{"R hip_p",""}, {"R hip_r",""}, {"R hip_y",""},
//                                    {"R knee",""}, {"R ank_p",""}, {"R ank_r",""}};

//   // Figure dimensions
//   const int fw = 480, fh = 270;

//   while (!sim->exitrequest.load()) {
//     std::this_thread::sleep_for(std::chrono::milliseconds(33));  // ~30 Hz

//     if (!cmg_viz_enabled.load(std::memory_order_relaxed)) {
//       // If just disabled, clear figures once
//       if (!ring.empty()) {
//         ring.clear();
//         // Wait until render consumed previous batch
//         while (sim->newfigurerequest.load() != 0 && !sim->exitrequest.load()) {
//           std::this_thread::sleep_for(std::chrono::milliseconds(5));
//         }
//         sim->user_figures_new_.clear();
//         sim->newfigurerequest.store(1);
//       }
//       continue;
//     }

//     // Read shared memory
//     CMGVizData snap;
//     if (reader.read(snap)) {
//       CmgSample s;
//       std::memcpy(s.qref, snap.qref, sizeof(s.qref));
//       std::memcpy(s.actual, snap.actual_pos, sizeof(s.actual));
//       std::memcpy(s.command, snap.command, sizeof(s.command));
//       ring.push_back(s);
//       if (static_cast<int>(ring.size()) > MAX_RING) ring.pop_front();
//     }

//     if (ring.empty()) continue;

//     // Wait for render thread to consume previous batch
//     if (sim->newfigurerequest.load() != 0) continue;

//     // Build 3 figures
//     sim->user_figures_new_.clear();
//     sim->user_figures_new_.reserve(3);

//     {
//       mjrRect vp = {0, 2 * fh, fw, fh};  // top-left
//       mjvFigure fig;
//       BuildLegFigure(fig, ring, left_idx, N_LEG, "Left Leg (qref vs actual)", left_names);
//       sim->user_figures_new_.push_back({vp, fig});
//     }
//     {
//       mjrRect vp = {0, fh, fw, fh};  // middle-left
//       mjvFigure fig;
//       BuildLegFigure(fig, ring, right_idx, N_LEG, "Right Leg (qref vs actual)", right_names);
//       sim->user_figures_new_.push_back({vp, fig});
//     }
//     {
//       mjrRect vp = {0, 0, fw, fh};  // bottom-left
//       mjvFigure fig;
//       BuildCommandFigure(fig, ring);
//       sim->user_figures_new_.push_back({vp, fig});
//     }

//     sim->newfigurerequest.store(1);
//   }
// }

//---------------------------------------- CMG ghost overlay ----------------------------------------

void CmgGhostThread(mj::Simulate* sim) {
  CMGVizReader reader;
  mjData* d_ghost = nullptr;
  mjvScene ghost_scn;
  bool scene_ok = false;

  // USD env index → SDK motor index (= MuJoCo actuated joint index)
  // From deploy_scale_one.yaml joint_ids_map
  static const int usd_to_mj[29] = {
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
  };

  while (!sim->exitrequest.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(33));  // ~30 Hz

    if (!cmg_viz_enabled.load(std::memory_order_relaxed)) {
      // When disabled, clear ghost geoms once
      if (d_ghost && !sim->ghost_scn_geoms_.empty()) {
        while (sim->newghostrequest.load() != 0 && !sim->exitrequest.load())
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
        sim->ghost_scn_geoms_new_.clear();
        sim->newghostrequest.store(1);
      }
      continue;
    }

    // Wait for model to be loaded
    if (!m) continue;

    // Allocate ghost data & scene on first use (or after model reload)
    if (!d_ghost) {
      d_ghost = mj_makeData(m);
      if (!d_ghost) continue;
      mjv_makeScene(m, &ghost_scn, 5000);
      scene_ok = true;
    }

    // Read CMG data from shared memory
    CMGVizData snap;
    if (!reader.read(snap)) continue;

    // Wait for render to consume previous batch
    if (sim->newghostrequest.load() != 0) continue;

    // Copy base pose (free joint qpos[0:7]) from actual robot
    {
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
      if (d) {
        std::memcpy(d_ghost->qpos, d->qpos, 7 * sizeof(mjtNum));
      }
    }
    // Offset ghost -0.5 in x
    d_ghost->qpos[0] -= 0.5;

    // Set ghost joint positions from CMG qref (USD order → MuJoCo qpos)
    for (int u = 0; u < 29; u++) {
      d_ghost->qpos[7 + usd_to_mj[u]] = static_cast<mjtNum>(snap.qref[u]);
    }

    // Forward kinematics to compute body positions
    mj_forward(m, d_ghost);

    // Build ghost scene (dynamic bodies only — no floor/sky)
    ghost_scn.ngeom = 0;
    mjvOption ghost_opt;
    mjv_defaultOption(&ghost_opt);
    mjv_addGeoms(m, d_ghost, &ghost_opt, nullptr, mjCAT_DYNAMIC, &ghost_scn);

    // Copy geoms with transparent blue-ish tint
    std::vector<mjvGeom> geoms;
    geoms.reserve(ghost_scn.ngeom);
    for (int i = 0; i < ghost_scn.ngeom; i++) {
      mjvGeom g = ghost_scn.geoms[i];
      // Skip non-model geoms (decoration)
      if (g.objtype != mjOBJ_GEOM && g.objtype != mjOBJ_SITE) continue;
      g.rgba[0] = 0.2f;
      g.rgba[1] = 0.5f;
      g.rgba[2] = 0.9f;
      g.rgba[3] = 0.3f;
      geoms.push_back(g);
    }

    sim->ghost_scn_geoms_new_ = std::move(geoms);
    sim->newghostrequest.store(1);
  }

  if (d_ghost) mj_deleteData(d_ghost);
  if (scene_ok) mjv_freeScene(&ghost_scn);
}

//-------------------------------------- telemetry viz ---------------------------------------------

std::atomic<bool> telemetry_viz_enabled{false};

struct TelemetrySample {
  float vx, vy, vz;                // GT base linear velocity (body frame)
  float thermal_load[29];           // EMA of tau² — proportional to motor heating
};

void TelemetryVizThread(mj::Simulate* sim) {
  std::deque<TelemetrySample> ring;
  const int MAX_RING = 200;

  // Thermal load: exponential moving average of tau² (Nm²)
  // Directly proportional to I²R motor heating (tau ∝ current).
  // Rises under sustained load, decays when load is removed.
  const float tau_smooth = 3.0f;     // EMA time constant (seconds)
  float thermal_load[29] = {};
  double prev_sim_time = -1.0;
  double last_logged_sim_time = -1.0;

  // Figure dimensions
  const int fw = 480, fh = 220;

  // Open CSV log file with timestamped name
  std::ofstream csv;
  std::filesystem::path log_path;
  {
    std::filesystem::path log_dir =
      std::filesystem::path(getExecutableDir()).parent_path() / "logs";
    std::error_code ec;
    std::filesystem::create_directories(log_dir, ec);

    std::time_t now = std::time(nullptr);
    std::tm tm_buf;
    localtime_r(&now, &tm_buf);
    std::ostringstream fname;
    fname << "telemetry_" << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << ".csv";
    log_path = log_dir / fname.str();

    csv.open(log_path);
    if (csv.is_open()) {
      csv << "sim_time,vx_fwd";
      for (int i = 0; i < 29; i++) csv << ",load_" << i;
      csv << ",qpos_Lknee,qpos_Rknee,ctrl_Lknee,ctrl_Rknee,tau_Lknee,tau_Rknee";
      csv << ",qpos_LhipP,qpos_RhipP,qpos_LhipY,qpos_RhipY"
             ",qpos_LankP,qpos_RankP,qpos_LankR,qpos_RankR";
      // CMG qref and policy residual for the same 5 leg-joint pairs
      csv << ",qref_Lknee,qref_Rknee,qref_LhipP,qref_RhipP,qref_LhipY,qref_RhipY"
             ",qref_LankP,qref_RankP,qref_LankR,qref_RankR";
      csv << ",res_Lknee,res_Rknee,res_LhipP,res_RhipP,res_LhipY,res_RhipY"
             ",res_LankP,res_RankP,res_LankR,res_RankR";
      // Foot ground reaction force Z (world frame, from cfrc_ext)
      csv << ",fz_Lfoot,fz_Rfoot,seq_cmg";
      csv << "\n";
      csv.flush();
      std::printf("[TelemetryViz] logging to %s\n", log_path.c_str());
    } else {
      std::fprintf(stderr, "[TelemetryViz] failed to open %s\n", log_path.c_str());
    }
  }

  int flush_counter = 0;

  // CMG shared-memory reader: pulls qref + residual published by State_RLResidual.
  // read() returns false until policy is running; snap retains last known values.
  CMGVizReader cmg_reader;
  CMGVizData cmg_snap{};
  uint32_t last_cmg_seq = 0;

  // Foot body IDs (resolved once after model loads)
  int Lfoot_bid = -1, Rfoot_bid = -1;

  while (!sim->exitrequest.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(33));  // ~30 Hz

    if (!m || !d) continue;

    if (Lfoot_bid < 0) {
      const char* lnames[] = {"left_ankle_roll_link", "left_foot_link", "left_ankle_link"};
      const char* rnames[] = {"right_ankle_roll_link", "right_foot_link", "right_ankle_link"};
      for (auto n : lnames) { int b = mj_name2id(m, mjOBJ_BODY, n); if (b >= 0) { Lfoot_bid = b; break; } }
      for (auto n : rnames) { int b = mj_name2id(m, mjOBJ_BODY, n); if (b >= 0) { Rfoot_bid = b; break; } }
      std::printf("[TelemetryViz] foot body ids: L=%d R=%d\n", Lfoot_bid, Rfoot_bid);
    }

    // Refresh CMG snapshot if writer published a new sample
    cmg_reader.read(cmg_snap);

    TelemetrySample s = {};
    int num_joints = 0;
    double sim_time = 0.0;
    bool sample_valid = false;
    {
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
      if (!d) continue;

      // GT base linear velocity: world-frame qvel → body-frame via quaternion
      {
        mjtNum quat_conj[4] = {d->qpos[3], -d->qpos[4], -d->qpos[5], -d->qpos[6]};
        mjtNum v_world[3] = {d->qvel[0], d->qvel[1], d->qvel[2]};
        mjtNum v_body[3];
        mju_rotVecQuat(v_body, v_world, quat_conj);
        s.vx = static_cast<float>(v_body[0]);  // forward
        s.vy = static_cast<float>(v_body[1]);  // lateral
        s.vz = static_cast<float>(v_body[2]);  // vertical
      }

      num_joints = std::min(m->nu, 29);

      // Compute dt from simulation time (handles speed changes & pauses)
      sim_time = d->time;
      float dt = 0.0f;
      if (prev_sim_time < 0.0 || sim_time < prev_sim_time) {
        // First tick or simulation was reset
        for (int i = 0; i < 29; i++) thermal_load[i] = 0.0f;
        last_logged_sim_time = -1.0;
        dt = 0.0f;
      } else {
        dt = static_cast<float>(sim_time - prev_sim_time);
      }
      prev_sim_time = sim_time;

      // EMA of tau²: load += (tau² - load) * alpha,  alpha = 1 - exp(-dt/tau_smooth)
      float alpha = (dt > 0.0f) ? 1.0f - std::exp(-dt / tau_smooth) : 0.0f;
      for (int i = 0; i < num_joints; i++) {
        float tau = static_cast<float>(d->sensordata[i + 2 * m->nu]);
        thermal_load[i] += (tau * tau - thermal_load[i]) * alpha;
        s.thermal_load[i] = thermal_load[i];
      }
      sample_valid = true;
    }

    // Always log to CSV (skip duplicates when sim is paused).
    // Build the entire row in an ostringstream and write atomically with flush —
    // prevents partial rows when the process is killed mid-row.
    if (sample_valid && csv.is_open() && sim_time != last_logged_sim_time) {
      std::ostringstream row;
      row << sim_time << ',' << s.vx;
      for (int i = 0; i < 29; i++) row << ',' << s.thermal_load[i];
      {
        const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
        if (d && m) {
          int lk = 3, rk = 9;  // L/R knee joint indices
          row << ',' << d->sensordata[lk]
              << ',' << d->sensordata[rk]
              << ',' << d->ctrl[lk]
              << ',' << d->ctrl[rk]
              << ',' << d->sensordata[lk + 2 * m->nu]
              << ',' << d->sensordata[rk + 2 * m->nu];
          // L/R hip pitch (0,6), hip yaw (2,8), ankle pitch (4,10), ankle roll (5,11)
          int idx[8] = {0, 6, 2, 8, 4, 10, 5, 11};
          for (int k = 0; k < 8; k++) row << ',' << d->sensordata[idx[k]];

          // CMG qref + policy residual in USD joint order
          // L/R: knee=9/10, hipP=0/1, hipY=6/7, ankP=13/14, ankR=17/18
          int usd_idx[10] = {9, 10, 0, 1, 6, 7, 13, 14, 17, 18};
          for (int k = 0; k < 10; k++) row << ',' << cmg_snap.qref[usd_idx[k]];
          for (int k = 0; k < 10; k++) row << ',' << cmg_snap.raw_residual[usd_idx[k]];

          // Foot Z ground reaction force (world frame, cfrc_ext layout: [tx,ty,tz,fx,fy,fz])
          double fzL = (Lfoot_bid >= 0) ? d->cfrc_ext[6 * Lfoot_bid + 5] : 0.0;
          double fzR = (Rfoot_bid >= 0) ? d->cfrc_ext[6 * Rfoot_bid + 5] : 0.0;
          row << ',' << fzL << ',' << fzR << ',' << cmg_snap.seq.load(std::memory_order_relaxed);
        } else {
          row << ",0,0,0,0,0,0";
          row << ",0,0,0,0,0,0,0,0";
          row << ",0,0,0,0,0,0,0,0,0,0";  // qref
          row << ",0,0,0,0,0,0,0,0,0,0";  // res
          row << ",0,0,0";                  // fz_L,fz_R,seq
        }
      }
      row << '\n';
      csv << row.str() << std::flush;
      last_logged_sim_time = sim_time;
      flush_counter = 0;  // each row already flushes; counter kept for compat
    }

    // Figures are only rendered when viz toggle is on
    if (!telemetry_viz_enabled.load(std::memory_order_relaxed)) {
      if (!ring.empty()) {
        ring.clear();
        while (sim->newfigurerequest.load() != 0 && !sim->exitrequest.load())
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
        sim->user_figures_new_.clear();
        sim->newfigurerequest.store(1);
      }
      continue;
    }

    ring.push_back(s);
    if (static_cast<int>(ring.size()) > MAX_RING) ring.pop_front();
    if (ring.empty()) continue;

    // Wait for render thread to consume previous batch
    if (sim->newfigurerequest.load() != 0) continue;

    sim->user_figures_new_.clear();
    sim->user_figures_new_.reserve(2);

    // --- Figure 1: GT Linear Velocity ---
    {
      mjrRect vp = {0, fh, fw, fh};  // top-left
      mjvFigure fig;
      mjv_defaultFigure(&fig);
      fig.flg_extend = 1;
      fig.flg_legend = 1;
      snprintf(fig.title, sizeof(fig.title), "GT Base Velocity - Body Frame (m/s)");
      snprintf(fig.yformat, sizeof(fig.yformat), "%%.1f");

      static const char* names[] = {"fwd", "lat", "vert"};
      static const float colors[][3] = {
        {1.0f, 0.3f, 0.3f},  // red
        {0.3f, 1.0f, 0.3f},  // green
        {0.3f, 0.3f, 1.0f},  // blue
      };

      int n_pts = static_cast<int>(ring.size());
      if (n_pts > 1000) n_pts = 1000;

      for (int i = 0; i < 3; i++) {
        snprintf(fig.linename[i], sizeof(fig.linename[0]), "%s", names[i]);
        for (int c = 0; c < 3; c++) fig.linergb[i][c] = colors[i][c];
        fig.linepnt[i] = n_pts;
        for (int t = 0; t < n_pts; t++) {
          const auto& sample = ring[ring.size() - n_pts + t];
          fig.linedata[i][2 * t]     = static_cast<float>(t);
          fig.linedata[i][2 * t + 1] = (i == 0) ? sample.vx : (i == 1) ? sample.vy : sample.vz;
        }
      }
      for (int l = 3; l < mjMAXLINE; l++) fig.linepnt[l] = 0;

      sim->user_figures_new_.push_back({vp, fig});
    }

    // --- Figure 2: Joint Temperatures ---
    {
      mjrRect vp = {0, 0, fw, fh};  // bottom-left
      mjvFigure fig;
      mjv_defaultFigure(&fig);
      fig.flg_extend = 1;
      fig.flg_legend = 1;
      snprintf(fig.title, sizeof(fig.title), "Motor Thermal Load (Nm^2)");
      snprintf(fig.yformat, sizeof(fig.yformat), "%%.0f");

      // All 12 leg joints: 6 left + 6 right
      static const int joint_idx[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
      static const char* joint_names[] = {
        "L hip_p", "L hip_r", "L hip_y", "L knee", "L ank_p", "L ank_r",
        "R hip_p", "R hip_r", "R hip_y", "R knee", "R ank_p", "R ank_r",
      };
      static const float hues[][3] = {
        {1.0f, 0.2f, 0.2f}, {0.2f, 0.8f, 0.2f}, {0.3f, 0.3f, 1.0f},
        {1.0f, 0.8f, 0.0f}, {1.0f, 0.4f, 0.8f}, {0.0f, 0.8f, 0.8f},
        {0.8f, 0.1f, 0.1f}, {0.1f, 0.6f, 0.1f}, {0.2f, 0.2f, 0.8f},
        {0.8f, 0.6f, 0.0f}, {0.8f, 0.3f, 0.6f}, {0.0f, 0.6f, 0.6f},
      };
      const int n_lines = 12;
      int n_pts = static_cast<int>(ring.size());
      if (n_pts > 1000) n_pts = 1000;

      for (int i = 0; i < n_lines; i++) {
        snprintf(fig.linename[i], sizeof(fig.linename[0]), "%s", joint_names[i]);
        for (int c = 0; c < 3; c++) fig.linergb[i][c] = hues[i][c];
        fig.linepnt[i] = n_pts;
        for (int t = 0; t < n_pts; t++) {
          const auto& sample = ring[ring.size() - n_pts + t];
          fig.linedata[i][2 * t]     = static_cast<float>(t);
          fig.linedata[i][2 * t + 1] = sample.thermal_load[joint_idx[i]];
        }
      }
      for (int l = n_lines; l < mjMAXLINE; l++) fig.linepnt[l] = 0;

      sim->user_figures_new_.push_back({vp, fig});
    }

    sim->newfigurerequest.store(1);
  }

  if (csv.is_open()) {
    csv.flush();
    csv.close();
    std::printf("[TelemetryViz] saved log: %s\n", log_path.c_str());
  }
}

//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char *msg)
{
  rosetta_error_msg = msg;
}
#endif

// user keyboard callback
void user_key_cb(GLFWwindow* window, int key, int scancode, int act, int mods) {
  if (act==GLFW_PRESS)
  {
    if(param::config.enable_elastic_band == 1) {
      if (key==GLFW_KEY_9) {
        elastic_band.enable_ = !elastic_band.enable_;
      } else if (key==GLFW_KEY_7 || key==GLFW_KEY_UP) {
        elastic_band.length_ -= 0.1;
      } else if (key==GLFW_KEY_8 || key==GLFW_KEY_DOWN) {
        elastic_band.length_ += 0.1;
      }
    }
    if(key==GLFW_KEY_BACKSPACE) {
      mj_resetData(m, d);
      mj_forward(m, d);
    }
    if(key==GLFW_KEY_C) {
      bool prev = cmg_viz_enabled.load();
      cmg_viz_enabled.store(!prev);
      std::printf("[CMGViz] %s\n", prev ? "OFF" : "ON");
    }
    if(key==GLFW_KEY_T) {
      bool prev = telemetry_viz_enabled.load();
      telemetry_viz_enabled.store(!prev);
      std::printf("[TelemetryViz] %s\n", prev ? "OFF" : "ON");
    }
  }
}

// run event loop
int main(int argc, char **argv)
{

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg)
  {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  // Always show contact points on viewer open
  opt.flags[mjVIS_CONTACTPOINT] = 1;

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // Load simulation configuration
  std::filesystem::path proj_dir = std::filesystem::path(getExecutableDir()).parent_path();
  param::config.load_from_yaml(proj_dir / "config.yaml");
  param::helper(argc, argv);
  if(param::config.robot_scene.is_relative()) {
    param::config.robot_scene = proj_dir.parent_path() / "unitree_robots" / param::config.robot / param::config.robot_scene;
  }

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>(
    std::make_unique<mj::GlfwAdapter>(),
    &cam, &opt, &pert, /* is_passive = */ false);

  std::thread unitree_thread(UnitreeSdk2BridgeThread, nullptr);

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), param::config.robot_scene.c_str());
  // start CMG visualization thread
  // std::thread cmgvizhandle(&CmgVizThread, sim.get());
  // start CMG ghost overlay thread
  std::thread cmgghosthandle(&CmgGhostThread, sim.get());
  // start telemetry visualization thread
  std::thread telemetryvizhandle(&TelemetryVizThread, sim.get());
  // start simulation UI loop (blocking call)
  glfwSetKeyCallback(static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_,user_key_cb);
  sim->RenderLoop();
  physicsthreadhandle.join();
  // cmgvizhandle.join();
  cmgghosthandle.join();
  telemetryvizhandle.join();

  pthread_exit(NULL);
  return 0;
}

// Quick benchmark: MLX vs CPU brute-force vs OpenGL matching.
// Build: ninja colmap_sift_mlx_bench
// Run:   ./src/colmap/feature/sift_mlx_bench

#include "colmap/feature/sift.h"
#include "colmap/feature/sift_mlx.h"
#include "colmap/feature/types.h"
#include "colmap/util/logging.h"
#include "colmap/util/opengl_utils.h"

#if defined(COLMAP_GUI_ENABLED)
#include <QApplication>
#endif

#include <chrono>
#include <iostream>
#include <random>

using namespace colmap;

FeatureDescriptorsData MakeRandomDescriptors(int num_features, int seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  FeatureDescriptorsData desc(num_features, 128);
  for (int i = 0; i < num_features; ++i) {
    for (int j = 0; j < 128; ++j) {
      desc(i, j) = static_cast<uint8_t>(dist(rng));
    }
  }
  return desc;
}

// Create two descriptor sets with known matches: desc2 is desc1 with noise.
// This ensures all matchers find real matches and do full work.
std::pair<FeatureDescriptorsData, FeatureDescriptorsData>
MakeMatchableDescriptors(int num_features, int seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> base_dist(10, 245);
  std::uniform_int_distribution<int> noise_dist(-5, 5);

  FeatureDescriptorsData desc1(num_features, 128);
  FeatureDescriptorsData desc2(num_features, 128);
  for (int i = 0; i < num_features; ++i) {
    for (int j = 0; j < 128; ++j) {
      int val = base_dist(rng);
      desc1(i, j) = static_cast<uint8_t>(val);
      desc2(i, j) = static_cast<uint8_t>(
          std::clamp(val + noise_dist(rng), 0, 255));
    }
  }
  return {desc1, desc2};
}

FeatureKeypoints MakeRandomKeypoints(int num_features, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
  FeatureKeypoints kps(num_features);
  for (int i = 0; i < num_features; ++i) {
    kps[i] = FeatureKeypoint(dist(rng), dist(rng));
  }
  return kps;
}

void BenchmarkMatcher(const std::string& name,
                      FeatureMatcher* matcher,
                      FeatureMatcher::Image img1,
                      FeatureMatcher::Image img2,
                      int num_warmup,
                      int num_iters) {
  FeatureMatches matches;

  // Warmup (use unique IDs to force descriptor upload).
  for (int i = 0; i < num_warmup; ++i) {
    img1.image_id = static_cast<image_t>(1000 + i * 2);
    img2.image_id = static_cast<image_t>(1000 + i * 2 + 1);
    matcher->Match(img1, img2, &matches);
  }

  // Timed runs — use different image_id each iteration to defeat
  // descriptor caching in OpenGL SiftGPU, ensuring full upload + match.
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iters; ++i) {
    img1.image_id = static_cast<image_t>(i * 2);
    img2.image_id = static_cast<image_t>(i * 2 + 1);
    matcher->Match(img1, img2, &matches);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  double avg_ms = total_ms / num_iters;

  std::cout << "  " << name << ": " << avg_ms << " ms/match ("
            << matches.size() << " matches, " << num_iters << " iters)"
            << std::endl;
}

struct BenchResult {
  std::string name;
  double ms;
};

std::vector<BenchResult> g_results;

void RunBenchmark(int num_features,
                  FeatureMatcher* gpu_matcher) {
  std::cout << "\n=== " << num_features << " features per image ==="
            << std::endl;

  auto [raw_desc1, raw_desc2] = MakeMatchableDescriptors(num_features, 42);
  auto desc1 = std::make_shared<FeatureDescriptors>(
      FeatureExtractorType::SIFT, std::move(raw_desc1));
  auto desc2 = std::make_shared<FeatureDescriptors>(
      FeatureExtractorType::SIFT, std::move(raw_desc2));
  auto kp1 = std::make_shared<FeatureKeypoints>(
      MakeRandomKeypoints(num_features, 42));
  auto kp2 = std::make_shared<FeatureKeypoints>(
      MakeRandomKeypoints(num_features, 123));

  FeatureMatcher::Image img1;
  img1.image_id = 1;
  img1.descriptors = desc1;
  img1.keypoints = kp1;

  FeatureMatcher::Image img2;
  img2.image_id = 2;
  img2.descriptors = desc2;
  img2.keypoints = kp2;

  int num_warmup = 2;
  int num_iters = num_features <= 2000 ? 20 : 10;

  // CPU brute-force.
  {
    auto sift_opts = std::make_shared<SiftMatchingOptions>();
    sift_opts->cpu_brute_force_matcher = true;
    FeatureMatchingOptions opts(FeatureMatcherType::SIFT_BRUTEFORCE);
    opts.sift = sift_opts;
    opts.use_gpu = false;
    auto matcher = CreateSiftFeatureMatcher(opts);
    if (matcher) {
      BenchmarkMatcher("CPU brute-force", matcher.get(),
                       img1, img2, num_warmup, num_iters);
    }
  }

  // OpenGL GPU.
  if (gpu_matcher) {
    BenchmarkMatcher("OpenGL GPU", gpu_matcher,
                     img1, img2, num_warmup, num_iters);
  }

  // MLX (Metal GPU).
#if defined(COLMAP_MLX_ENABLED)
  {
    auto sift_opts = std::make_shared<SiftMatchingOptions>();
    FeatureMatchingOptions opts(FeatureMatcherType::SIFT_BRUTEFORCE);
    opts.sift = sift_opts;
    opts.use_gpu = true;
    opts.gpu_index = "0";
    auto matcher = CreateSiftMLXFeatureMatcher(opts);
    if (matcher) {
      BenchmarkMatcher("MLX (Metal GPU)", matcher.get(),
                       img1, img2, num_warmup, num_iters);
    }
  }
#else
  std::cout << "  MLX: not available" << std::endl;
#endif
}

void RunAllBenchmarks(FeatureMatcher* gpu_matcher) {
  std::cout << "SIFT Feature Matching Benchmark" << std::endl;
  std::cout << "================================" << std::endl;

  RunBenchmark(500, gpu_matcher);
  RunBenchmark(1000, gpu_matcher);
  RunBenchmark(2000, gpu_matcher);
  RunBenchmark(4000, gpu_matcher);
  RunBenchmark(8192, gpu_matcher);
}

int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);

#if defined(COLMAP_GPU_ENABLED) && defined(COLMAP_GUI_ENABLED)
  // Need QApplication + OpenGL context for the SiftGPU matcher.
  QApplication app(argc, argv);

  class BenchThread : public Thread {
   public:
    void Run() override {
      opengl_context_.MakeCurrent();

      // Create the OpenGL GPU matcher.
      auto sift_opts = std::make_shared<SiftMatchingOptions>();
      FeatureMatchingOptions opts(FeatureMatcherType::SIFT_BRUTEFORCE);
      opts.sift = sift_opts;
      opts.use_gpu = true;
      opts.gpu_index = "0";
      auto matcher = CreateSiftFeatureMatcher(opts);

      RunAllBenchmarks(matcher.get());
    }
    OpenGLContextManager opengl_context_;
  };

  BenchThread thread;
  RunThreadWithOpenGLContext(&thread);
#else
  std::cout << "(OpenGL GPU matcher not available)" << std::endl;
  RunAllBenchmarks(nullptr);
#endif

  return 0;
}

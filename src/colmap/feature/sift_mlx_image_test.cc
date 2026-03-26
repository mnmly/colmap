// CPU vs MLX SIFT matching comparison on real images.
//
// Build: ninja colmap_sift_mlx_image_test
// Run:   ./src/colmap/feature/sift_mlx_image_test /path/to/images
//
// Extracts SIFT features from every image in the directory, then runs all
// N*(N-1)/2 pairs through both the CPU brute-force matcher and the MLX custom-
// kernel matcher.  Reports:
//   - per-pair match counts (CPU vs MLX)
//   - any pairs where the match sets differ
//   - a summary table

#include "colmap/feature/extractor.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/sift_mlx.h"
#include "colmap/feature/types.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace colmap;

// ---- helpers ---------------------------------------------------------------

static bool IsImageFile(const fs::path& p) {
  const std::string ext = p.extension().string();
  for (const auto& e : {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
                        ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"}) {
    if (ext == e) return true;
  }
  return false;
}

// Collect (sorted) image paths from a directory.
static std::vector<fs::path> CollectImages(const fs::path& dir) {
  std::vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file() && IsImageFile(entry.path())) {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

// Sort matches by (idx1, idx2) for stable comparison.
static void SortMatches(FeatureMatches& m) {
  std::sort(m.begin(), m.end(), [](const auto& a, const auto& b) {
    return a.point2D_idx1 < b.point2D_idx1 ||
           (a.point2D_idx1 == b.point2D_idx1 &&
            a.point2D_idx2 < b.point2D_idx2);
  });
}

static bool MatchesEqual(FeatureMatches a, FeatureMatches b) {
  SortMatches(a);
  SortMatches(b);
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].point2D_idx1 != b[i].point2D_idx1 ||
        a[i].point2D_idx2 != b[i].point2D_idx2)
      return false;
  }
  return true;
}

// ---- main ------------------------------------------------------------------

int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_dir> [--cross-check=0|1]\n";
    return 1;
  }

  const fs::path image_dir(argv[1]);
  if (!fs::is_directory(image_dir)) {
    std::cerr << "Not a directory: " << image_dir << "\n";
    return 1;
  }

  bool cross_check = true;
  for (int i = 2; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--cross-check=0") cross_check = false;
    if (arg == "--cross-check=1") cross_check = true;
  }

  // ---- Extract SIFT features -----------------------------------------------

  FeatureExtractionOptions ext_opts(FeatureExtractorType::SIFT);
  ext_opts.use_gpu = false;  // force CPU extraction
  auto extractor = CreateSiftFeatureExtractor(ext_opts);
  CHECK(extractor);

  struct ImageData {
    std::string name;
    FeatureMatcher::Image matcher_image;
    std::shared_ptr<FeatureKeypoints> keypoints;
    std::shared_ptr<FeatureDescriptors> descriptors;
  };

  std::vector<ImageData> images;
  const auto paths = CollectImages(image_dir);

  if (paths.empty()) {
    std::cerr << "No images found in " << image_dir << "\n";
    return 1;
  }

  std::cout << "Extracting SIFT features from " << paths.size()
            << " image(s)...\n";

  const Camera dummy_camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 100);

  const int max_image_size = ext_opts.EffMaxImageSize();

  for (size_t i = 0; i < paths.size(); ++i) {
    Bitmap bmp;
    if (!bmp.Read(paths[i], /*as_rgb=*/false)) {
      std::cerr << "  [SKIP] cannot read: " << paths[i].filename() << "\n";
      continue;
    }
    // Rescale if the image exceeds the extractor's maximum dimension.
    const int max_dim = std::max(bmp.Width(), bmp.Height());
    if (max_dim > max_image_size) {
      const float scale =
          static_cast<float>(max_image_size) / static_cast<float>(max_dim);
      bmp.Rescale(static_cast<int>(bmp.Width() * scale),
                  static_cast<int>(bmp.Height() * scale));
    }
    auto kps = std::make_shared<FeatureKeypoints>();
    auto desc = std::make_shared<FeatureDescriptors>();
    if (!extractor->Extract(bmp, kps.get(), desc.get())) {
      std::cerr << "  [SKIP] extraction failed: " << paths[i].filename()
                << "\n";
      continue;
    }
    std::cout << "  " << paths[i].filename().string() << ": "
              << kps->size() << " keypoints\n";

    ImageData d;
    d.name = paths[i].filename().string();
    d.keypoints = std::move(kps);
    d.descriptors = std::move(desc);
    d.matcher_image = {
        static_cast<image_t>(i),
        &dummy_camera,
        d.keypoints,
        d.descriptors,
    };
    images.push_back(std::move(d));
  }

  if (images.size() < 2) {
    std::cerr << "Need at least 2 images.\n";
    return 1;
  }

  // ---- Create matchers -------------------------------------------------------

  auto sift_opts = std::make_shared<SiftMatchingOptions>();
  sift_opts->cross_check = cross_check;
  sift_opts->cpu_brute_force_matcher = true;

  FeatureMatchingOptions bf_opts(FeatureMatcherType::SIFT_BRUTEFORCE);
  bf_opts.sift = sift_opts;
  bf_opts.use_gpu = false;
  auto cpu_matcher = CreateSiftFeatureMatcher(bf_opts);
  CHECK(cpu_matcher);

#if defined(COLMAP_MLX_ENABLED)
  FeatureMatchingOptions mlx_opts(FeatureMatcherType::SIFT_BRUTEFORCE);
  mlx_opts.sift = sift_opts;
  mlx_opts.use_gpu = true;
  auto mlx_matcher = CreateSiftMLXFeatureMatcher(mlx_opts);
  CHECK(mlx_matcher);
#else
  std::cerr << "MLX not available — exiting.\n";
  return 1;
#endif

  // ---- Match all pairs -------------------------------------------------------

  const int N = static_cast<int>(images.size());
  const int num_pairs = N * (N - 1) / 2;

  std::cout << "\nMatching " << num_pairs << " pair(s) "
            << "(cross_check=" << cross_check << ")...\n\n";

  // Print table header
  const int w = 22;
  std::cout << std::left
            << std::setw(w) << "Image A"
            << std::setw(w) << "Image B"
            << std::setw(10) << "CPU"
            << std::setw(10) << "MLX"
            << std::setw(10) << "Match?\n";
  std::cout << std::string(w * 2 + 30, '-') << "\n";

  int num_differ = 0;
  double total_cpu_ms = 0, total_mlx_ms = 0;

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      FeatureMatches m_cpu, m_mlx;

      auto t0 = std::chrono::high_resolution_clock::now();
      cpu_matcher->Match(images[i].matcher_image, images[j].matcher_image,
                         &m_cpu);
      auto t1 = std::chrono::high_resolution_clock::now();
      mlx_matcher->Match(images[i].matcher_image, images[j].matcher_image,
                         &m_mlx);
      auto t2 = std::chrono::high_resolution_clock::now();

      total_cpu_ms +=
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      total_mlx_ms +=
          std::chrono::duration<double, std::milli>(t2 - t1).count();

      const bool equal = MatchesEqual(m_cpu, m_mlx);
      if (!equal) ++num_differ;

      // Truncate long names for the table
      auto trunc = [](const std::string& s, int n) {
        return s.size() > static_cast<size_t>(n) ? s.substr(0, n - 1) + "…"
                                                  : s;
      };

      std::cout << std::left
                << std::setw(w) << trunc(images[i].name, w)
                << std::setw(w) << trunc(images[j].name, w)
                << std::setw(10) << m_cpu.size()
                << std::setw(10) << m_mlx.size()
                << (equal ? "OK" : "DIFFER") << "\n";

      if (!equal) {
        // Show first few differing entries
        FeatureMatches a = m_cpu, b = m_mlx;
        SortMatches(a);
        SortMatches(b);
        std::cout << "    CPU[0..4]:";
        for (size_t k = 0; k < std::min<size_t>(4, a.size()); ++k)
          std::cout << " (" << a[k].point2D_idx1 << "→" << a[k].point2D_idx2
                    << ")";
        std::cout << "\n    MLX[0..4]:";
        for (size_t k = 0; k < std::min<size_t>(4, b.size()); ++k)
          std::cout << " (" << b[k].point2D_idx1 << "→" << b[k].point2D_idx2
                    << ")";
        std::cout << "\n";
      }
    }
  }

  // ---- Summary ---------------------------------------------------------------

  std::cout << "\n" << std::string(w * 2 + 30, '=') << "\n";
  std::cout << "Pairs: " << num_pairs
            << "  |  Differ: " << num_differ
            << "  |  CPU total: " << std::fixed << std::setprecision(1)
            << total_cpu_ms << " ms"
            << "  |  MLX total: " << total_mlx_ms << " ms";
  if (total_cpu_ms > 0)
    std::cout << "  (speedup: " << std::setprecision(2)
              << total_cpu_ms / total_mlx_ms << "×)";
  std::cout << "\n";

  return num_differ > 0 ? 1 : 0;
}

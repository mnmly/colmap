// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/feature/sift_mlx.h"

#if defined(COLMAP_MLX_ENABLED)

#include "colmap/feature/sift.h"
#include "colmap/feature/sift_internal.h"
#include "colmap/util/logging.h"

#include <mutex>

#include <Eigen/Core>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace colmap {

namespace {

// Global mutex for thread safety since MLX Metal operations
// are not explicitly thread-safe for concurrent submissions.
std::mutex mlx_matcher_mutex;

// Zero-copy uint8 view of a descriptor matrix, valid until `descriptors`
// is destroyed.  eval() must be called before `descriptors` goes out of scope.
mx::array EigenUint8ToMLXUint8(const FeatureDescriptorsData& descriptors) {
  const int rows = static_cast<int>(descriptors.rows());
  const int cols = static_cast<int>(descriptors.cols());
  return mx::array(
      const_cast<void*>(static_cast<const void*>(descriptors.data())),
      {rows, cols},
      mx::uint8,
      [](void*) {});
}

// ---------------------------------------------------------------------------
// Custom Metal kernel: fused SIFT one-way matching (v3)
//
// One SIMD group (32 threads) per query descriptor.
//   - Each lane j streams through M/32 reference descriptors in strides of 32,
//     computing the dot product on the fly and maintaining a per-lane top-2.
//   - A SIMD shuffle-down tree reduces the 32 lanes into the global top-2
//     for that query without any threadgroup (shared) memory.
//   - Ratio test + angular distance filter run entirely on-chip.
//   - Only a (N,) int32 result array is written back — the N×M float
//     matrix is never allocated.
//
// Grid: (N*32, 1, 1) total threads → N threadgroups of 32 threads.
// MLX auto-generates the kernel signature from input/output array types and
// adds Metal attributes that appear in the source (threadgroup_position_in_grid,
// thread_index_in_threadgroup).
// ---------------------------------------------------------------------------

// clang-format off
static const char* kMatchKernelHeader = R"(
#include <metal_stdlib>
using namespace metal;
)";

static const char* kMatchKernelBody = R"(
  constexpr uint kT = 32u;                        // SIMD group width

  uint i = threadgroup_position_in_grid.x;    // query index 0..N-1
  uint t = thread_index_in_threadgroup;       // SIMD lane 0..31 (scalar uint)

  uint N = (uint)desc1_shape[0];
  uint M = (uint)desc2_shape[0];

  if (i >= N) return;

  const device uint8_t* q = desc1 + i * 128u;    // query descriptor row

  // Per-lane running top-2 over the candidates this lane is responsible for.
  float lane_best   = 0.0f;
  float lane_second = 0.0f;
  int   lane_idx    = -1;

  for (uint j = t; j < M; j += kT) {
    const device uint8_t* r = desc2 + j * 128u;
    float dot = 0.0f;
    for (uint d = 0; d < 128u; d++) {
      dot = fma(float(q[d]), float(r[d]), dot);
    }
    if (dot > lane_best) {
      lane_second = lane_best;
      lane_best   = dot;
      lane_idx    = (int)j;
    } else if (dot > lane_second) {
      lane_second = dot;
    }
  }

  // SIMD tree-reduction: merge 32 per-lane (best, second, idx) into one.
  // At each step the winner keeps its best and promotes the loser's best
  // as the new second if it beats the current second.
  for (uint stride = kT >> 1u; stride >= 1u; stride >>= 1u) {
    float o_best   = simd_shuffle_down(lane_best,   stride);
    float o_second = simd_shuffle_down(lane_second, stride);
    int   o_idx    = simd_shuffle_down(lane_idx,    stride);
    if (o_best > lane_best) {
      lane_second = max(lane_best,   o_second);
      lane_best   = o_best;
      lane_idx    = o_idx;
    } else {
      lane_second = max(lane_second, o_best);
    }
  }

  // Lane 0 holds the correct global top-2; apply ratio test and write output.
  if (t == 0u) {
    constexpr float kInvNorm = 1.0f / 262144.0f;  // 1 / (512*512)
    float da = acos(fmin(lane_best   * kInvNorm, 1.0f));
    float db = acos(fmin(lane_second * kInvNorm, 1.0f));
    matches[i] = (lane_idx >= 0 && da <= max_distance && da < max_ratio * db)
                     ? lane_idx : -1;
  }
)";
// clang-format on

// Returns the singleton CustomKernelFunction, compiled once on first use.
// MLX caches the resulting MTLLibrary by kernel name.
const mx::fast::CustomKernelFunction& SiftMatchKernel() {
  static const auto kKernel = mx::fast::metal_kernel(
      "sift_match_one_way",
      {"desc1", "desc2", "max_ratio", "max_distance"},
      {"matches"},
      kMatchKernelBody,
      kMatchKernelHeader);
  return kKernel;
}

// Dispatches the custom kernel for one matching direction.
// desc1: (N, 128) uint8 — queries
// desc2: (M, 128) uint8 — references
// Returns (N,) int32: best match index in [0, M) or -1 if none.
mx::array MatchOneWayCustomKernel(const mx::array& desc1,
                                  const mx::array& desc2,
                                  float max_ratio,
                                  float max_distance) {
  const int N = desc1.shape(0);
  constexpr int kSimdWidth = 32;
  mx::Shape out_shape = {N};
  // std::function doesn't forward default args — all 9 params are required.
  auto results = SiftMatchKernel()(
      {desc1, desc2, mx::array(max_ratio), mx::array(max_distance)},
      std::vector<mx::Shape>{out_shape},
      std::vector<mx::Dtype>{mx::int32},
      std::make_tuple(N * kSimdWidth, 1, 1),  // total threads: N groups × 32
      std::make_tuple(kSimdWidth, 1, 1),       // threadgroup size
      {},             // template_args (none)
      std::nullopt,   // init_value
      false,          // verbose
      {});            // default stream/device
  return results[0];
}

}  // namespace

class SiftMLXFeatureMatcher : public FeatureMatcher {
 public:
  explicit SiftMLXFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(options) {
    THROW_CHECK(options_.Check());
  }

  static std::unique_ptr<FeatureMatcher> Create(
      const FeatureMatchingOptions& options) {
    return std::make_unique<SiftMLXFeatureMatcher>(options);
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    internal::ThrowCheckFeatureTypesMatch(image1, image2);

    matches->clear();

    if (image1.descriptors->data.rows() == 0 ||
        image2.descriptors->data.rows() == 0) {
      return;
    }

    const int N = static_cast<int>(image1.descriptors->data.rows());
    const int M = static_cast<int>(image2.descriptors->data.rows());
    const float max_ratio = static_cast<float>(options_.sift->max_ratio);
    const float max_distance = static_cast<float>(options_.sift->max_distance);
    const bool cross_check = options_.sift->cross_check;

    mx::array matches_1to2 = mx::array(0);
    mx::array matches_2to1 = mx::array(0);

    {
      std::lock_guard<std::mutex> lock(mlx_matcher_mutex);

      // Zero-copy uint8 views — the Eigen matrices outlive this scope.
      auto desc1 = EigenUint8ToMLXUint8(image1.descriptors->data);
      auto desc2 = EigenUint8ToMLXUint8(image2.descriptors->data);

      // Fused top-2 + ratio test entirely on Metal GPU.
      // Only (N,) and optionally (M,) int32 arrays are transferred back.
      matches_1to2 =
          MatchOneWayCustomKernel(desc1, desc2, max_ratio, max_distance);
      if (cross_check) {
        matches_2to1 =
            MatchOneWayCustomKernel(desc2, desc1, max_ratio, max_distance);
        mx::eval(matches_1to2, matches_2to1);
      } else {
        mx::eval(matches_1to2);
      }
    }

    // O(N) CPU scan to collect (mutual) matches.
    const int32_t* m12 = matches_1to2.data<int32_t>();

    if (cross_check) {
      const int32_t* m21 = matches_2to1.data<int32_t>();
      for (int i = 0; i < N; ++i) {
        const int j = m12[i];
        if (j >= 0 && j < M && m21[j] == static_cast<int32_t>(i)) {
          matches->push_back(
              {static_cast<point2D_t>(i), static_cast<point2D_t>(j)});
        }
      }
    } else {
      for (int i = 0; i < N; ++i) {
        const int j = m12[i];
        if (j >= 0) {
          matches->push_back(
              {static_cast<point2D_t>(i), static_cast<point2D_t>(j)});
        }
      }
    }
  }

  void MatchGuided(const double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override {
    THROW_CHECK_NOTNULL(two_view_geometry);
    internal::ThrowCheckFeatureTypesMatch(
        image1, image2, /*check_keypoints=*/true);

    two_view_geometry->inlier_matches.clear();

    // For calibrated cases, use the essential matrix with normalized
    // coordinates.
    const bool use_essential_matrix =
        (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
         two_view_geometry->config == TwoViewGeometry::CALIBRATED_RIG) &&
        two_view_geometry->E.has_value();
    const bool use_fundamental_matrix =
        two_view_geometry->config == TwoViewGeometry::UNCALIBRATED &&
        two_view_geometry->F.has_value();
    const bool use_homography =
        (two_view_geometry->config == TwoViewGeometry::PLANAR ||
         two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
         two_view_geometry->config == TwoViewGeometry::PLANAR_OR_PANORAMIC) &&
        two_view_geometry->H.has_value();

    const FeatureKeypoints normalized_keypoints1 =
        use_essential_matrix
            ? internal::NormalizeFeatureKeypoints(
                  *image1.camera, *image1.keypoints)
            : FeatureKeypoints();
    const FeatureKeypoints normalized_keypoints2 =
        use_essential_matrix
            ? internal::NormalizeFeatureKeypoints(
                  *image2.camera, *image2.keypoints)
            : FeatureKeypoints();

    const Eigen::Matrix3f E_or_F =
        use_essential_matrix
            ? Eigen::Matrix3f(two_view_geometry->E->cast<float>())
        : use_fundamental_matrix
            ? Eigen::Matrix3f(two_view_geometry->F->cast<float>())
            : Eigen::Matrix3f::Zero();
    const Eigen::Matrix3f H =
        use_homography ? Eigen::Matrix3f(two_view_geometry->H->cast<float>())
                       : Eigen::Matrix3f::Zero();

    const float max_residual =
        use_essential_matrix
            ? static_cast<float>(
                  internal::ComputeNormalizedGuidedMatchingMaxResidual(
                      *image1.camera, *image2.camera, max_error))
            : static_cast<float>(max_error * max_error);

    std::function<bool(float, float, float, float)> guided_filter;
    if (use_essential_matrix || use_fundamental_matrix) {
      guided_filter = [&](const float x1,
                          const float y1,
                          const float x2,
                          const float y2) {
        const Eigen::Vector3f p1(x1, y1, 1.0f);
        const Eigen::Vector3f p2(x2, y2, 1.0f);
        const Eigen::Vector3f epipolar_line1 = E_or_F * p1;
        const Eigen::Vector3f epipolar_line2 = E_or_F.transpose() * p2;
        const float nom = p2.transpose() * epipolar_line1;
        const float denom_sq = epipolar_line1(0) * epipolar_line1(0) +
                               epipolar_line1(1) * epipolar_line1(1) +
                               epipolar_line2(0) * epipolar_line2(0) +
                               epipolar_line2(1) * epipolar_line2(1);
        return nom * nom > max_residual * denom_sq;
      };
    } else if (use_homography) {
      guided_filter = [&](const float x1,
                          const float y1,
                          const float x2,
                          const float y2) {
        const Eigen::Vector3f p1(x1, y1, 1.0f);
        const Eigen::Vector2f p2(x2, y2);
        return ((H * p1).hnormalized() - p2).squaredNorm() > max_residual;
      };
    } else {
      return;
    }

    THROW_CHECK(guided_filter);

    // Guided matching uses L2 distance with geometric filtering.
    // CPU fallback since geometric filtering is per-element.
    const Eigen::RowMajorMatrixXf l2_dists_1to2 =
        internal::ComputeSiftDistanceMatrix(
            internal::DistanceType::L2,
            use_essential_matrix ? &normalized_keypoints1
                                : image1.keypoints.get(),
            use_essential_matrix ? &normalized_keypoints2
                                : image2.keypoints.get(),
            image1.descriptors->data,
            image2.descriptors->data,
            guided_filter);
    const Eigen::RowMajorMatrixXf l2_dists_2to1 = l2_dists_1to2.transpose();

    Eigen::RowMajorMatrixXi indices_1to2(l2_dists_1to2.rows(),
                                         l2_dists_1to2.cols());
    for (int i = 0; i < indices_1to2.rows(); ++i) {
      indices_1to2.row(i) = Eigen::VectorXi::LinSpaced(
          indices_1to2.cols(), 0, indices_1to2.cols() - 1);
    }
    Eigen::RowMajorMatrixXi indices_2to1(l2_dists_1to2.cols(),
                                         l2_dists_1to2.rows());
    for (int i = 0; i < indices_2to1.rows(); ++i) {
      indices_2to1.row(i) = Eigen::VectorXi::LinSpaced(
          indices_2to1.cols(), 0, indices_2to1.cols() - 1);
    }

    internal::FindBestMatchesIndex(indices_1to2,
                                   l2_dists_1to2,
                                   indices_2to1,
                                   l2_dists_2to1,
                                   options_.sift->max_ratio,
                                   options_.sift->max_distance,
                                   options_.sift->cross_check,
                                   &two_view_geometry->inlier_matches);
  }

 private:
  const FeatureMatchingOptions options_;
};

std::unique_ptr<FeatureMatcher> CreateSiftMLXFeatureMatcher(
    const FeatureMatchingOptions& options) {
  LOG(INFO) << "Creating SIFT MLX feature matcher (Apple Silicon GPU)";
  return SiftMLXFeatureMatcher::Create(options);
}

}  // namespace colmap

#endif  // COLMAP_MLX_ENABLED

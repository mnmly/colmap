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

mx::array EigenUint8ToMLXFloat(const FeatureDescriptorsData& descriptors) {
  const int num_descriptors = static_cast<int>(descriptors.rows());
  const int dim = static_cast<int>(descriptors.cols());

  // Create MLX array from raw pointer (zero-copy for the uint8 data).
  // The no-op deleter means the Eigen matrix must outlive this array
  // until eval() is called.
  auto mlx_uint8 = mx::array(
      const_cast<void*>(static_cast<const void*>(descriptors.data())),
      {num_descriptors, dim},
      mx::uint8,
      [](void*) {});

  // Cast to float32 for matmul. This triggers a copy to GPU.
  return mx::astype(mlx_uint8, mx::float32);
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

    Eigen::RowMajorMatrixXf dot_products;

    {
      std::lock_guard<std::mutex> lock(mlx_matcher_mutex);

      auto desc1 = EigenUint8ToMLXFloat(image1.descriptors->data);
      auto desc2 = EigenUint8ToMLXFloat(image2.descriptors->data);

      // Compute pairwise dot products on Metal GPU:
      // dot_products[i][j] = desc1[i] . desc2[j]
      auto dots = mx::matmul(desc1, mx::transpose(desc2));
      mx::eval(dots);

      // Map result back to Eigen matrix.
      const int rows = static_cast<int>(image1.descriptors->data.rows());
      const int cols = static_cast<int>(image2.descriptors->data.rows());
      dot_products.resize(rows, cols);
      const float* data_ptr = dots.data<float>();
      std::copy(data_ptr, data_ptr + rows * cols, dot_products.data());
    }

    // Reuse existing CPU brute-force matching (ratio test + cross-check).
    internal::FindBestMatchesBruteForce(dot_products,
                                        options_.sift->max_ratio,
                                        options_.sift->max_distance,
                                        options_.sift->cross_check,
                                        matches);
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

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

#pragma once

#include "colmap/feature/matcher.h"
#include "colmap/feature/types.h"
#include "colmap/sensor/models.h"
#include "colmap/util/types.h"

#include <functional>

#include <Eigen/Core>

namespace colmap {
namespace internal {

constexpr int kSiftDescriptorDim = 128;
constexpr int kSqSiftDescriptorNorm = 512 * 512;

void ThrowCheckFeatureTypesMatch(const FeatureMatcher::Image& image1,
                                 const FeatureMatcher::Image& image2,
                                 bool check_keypoints = false);

size_t FindBestMatchesOneWayBruteForce(
    const Eigen::RowMajorMatrixXf& dot_products,
    float max_ratio,
    float max_distance,
    std::vector<int>* matches);

void FindBestMatchesBruteForce(const Eigen::RowMajorMatrixXf& dot_products,
                                float max_ratio,
                                float max_distance,
                                bool cross_check,
                                FeatureMatches* matches);

enum class DistanceType {
  L2,
  DOT_PRODUCT,
};

Eigen::RowMajorMatrixXf ComputeSiftDistanceMatrix(
    DistanceType distance_type,
    const FeatureKeypoints* keypoints1,
    const FeatureKeypoints* keypoints2,
    const FeatureDescriptorsData& descriptors1,
    const FeatureDescriptorsData& descriptors2,
    const std::function<bool(float, float, float, float)>& guided_filter);

FeatureKeypoints NormalizeFeatureKeypoints(const Camera& camera,
                                           const FeatureKeypoints& keypoints);

double ComputeNormalizedGuidedMatchingMaxResidual(const Camera& camera1,
                                                   const Camera& camera2,
                                                   double max_error);

size_t FindBestMatchesOneWayIndex(const Eigen::RowMajorMatrixXi& indices,
                                   const Eigen::RowMajorMatrixXf& l2_dists,
                                   float max_ratio,
                                   float max_distance,
                                   std::vector<int>* matches);

void FindBestMatchesIndex(const Eigen::RowMajorMatrixXi& indices_1to2,
                           const Eigen::RowMajorMatrixXf& l2_dists_1to2,
                           const Eigen::RowMajorMatrixXi& indices_2to1,
                           const Eigen::RowMajorMatrixXf& l2_dists_2to1,
                           float max_ratio,
                           float max_distance,
                           bool cross_check,
                           FeatureMatches* matches);

}  // namespace internal
}  // namespace colmap

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_RELAX_ANALYSIS_DNNFUSION_FUSIBILITY_H_
#define TVM_RELAX_ANALYSIS_DNNFUSION_FUSIBILITY_H_

#include "./dnnfusion_op_classifier.h"

namespace tvm {
namespace relax {

struct DnnfFuseDecision {
  bool allow{false};
  std::string reason;
  bool needs_depend{false};
};

DnnfFuseDecision CanFuseDnnf(const DnnfClassification& src,
                             const DnnfClassification& sink);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_DNNFUSION_FUSIBILITY_H_

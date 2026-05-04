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

#include "./dnnfusion_fusibility.h"

namespace tvm {
namespace relax {

namespace {

DnnfFuseDecision Allow(const std::string& reason) {
  return DnnfFuseDecision{true, reason, false};
}

DnnfFuseDecision Reject(const std::string& reason, bool needs_depend = false) {
  return DnnfFuseDecision{false, reason, needs_depend};
}

}  // namespace

DnnfFuseDecision CanFuseDnnf(const DnnfClassification& src,
                             const DnnfClassification& sink) {
  if (src.cls == DnnfOpClass::kUnknown || sink.cls == DnnfOpClass::kUnknown) {
    return Reject("unknown DNNFusion class");
  }

  if (src.cls == DnnfOpClass::kShuffle || sink.cls == DnnfOpClass::kShuffle) {
    return Reject("fuse_depend_unimplemented", true);
  }

  if (src.cls == DnnfOpClass::kManyToMany && sink.cls == DnnfOpClass::kManyToMany) {
    return Reject("many-to-many to many-to-many requires fuse_depend", true);
  }

  if (src.cls == DnnfOpClass::kReorganize || sink.cls == DnnfOpClass::kReorganize) {
    return Reject("reorganize fusion requires dependence-aware validation", true);
  }

  if (src.cls == DnnfOpClass::kOneToOne || sink.cls == DnnfOpClass::kOneToOne) {
    return Allow("one-to-one adjacent fusion");
  }

  if (src.cls == DnnfOpClass::kOneToMany && sink.cls == DnnfOpClass::kOneToMany) {
    return Allow("one-to-many adjacent fusion");
  }

  return Reject("DNNFusion class pair not enabled in exp3");
}

}  // namespace relax
}  // namespace tvm

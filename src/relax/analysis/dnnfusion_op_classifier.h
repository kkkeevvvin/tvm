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

/*!
 * \file src/relax/analysis/dnnfusion_op_classifier.h
 * \brief Internal helpers for DNNFusion op classification.
 *
 * Public types (DnnfOpClass, DnnfClassSource, DnnfClassification) live in
 * include/tvm/relax/dnnfusion_op_class.h. Public functions (ClassifyDnnfOp,
 * InferDnnfClassFromTir, DnnfClassFromOpPattern) are declared in
 * include/tvm/relax/analysis.h. This header carries only file-local glue
 * shared between the classifier .cc and the annotation pass .cc.
 */
#ifndef TVM_RELAX_ANALYSIS_DNNFUSION_OP_CLASSIFIER_H_
#define TVM_RELAX_ANALYSIS_DNNFUSION_OP_CLASSIFIER_H_

#include <tvm/relax/dnnfusion_op_class.h>

#include <string>

namespace tvm {
namespace relax {

/*! \brief Human-readable name for a DnnfOpClass — used in tests and dumps. */
std::string DnnfOpClassToString(DnnfOpClass cls);
/*! \brief Human-readable name for a DnnfClassSource. */
std::string DnnfClassSourceToString(DnnfClassSource src);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_DNNFUSION_OP_CLASSIFIER_H_

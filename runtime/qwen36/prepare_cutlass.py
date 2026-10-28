"""针对固定 v1.1.4 构建 SM89 使用的 CUTLASS NN/NT 路径；不修改训练源码。"""
from pathlib import Path
import hashlib
import json
import subprocess
root = Path('.deps-qwen36/src/grouped_gemm')
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == '172fada89fa7364fe5d026b3a0dfab58b591ffdd'
assert subprocess.check_output(['git', '-C', str(root/'third_party/cutlass'), 'rev-parse', 'HEAD'], text=True).strip() == '8783c41851cd3582490e04e69e0cd756a8c1db7f'
p = root/'csrc/grouped_gemm.cu'
s = subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:csrc/grouped_gemm.cu'], text=True)
s = s.replace('using GroupedGemmKernelNN = typename', 'template <typename LayoutB>\nusing GroupedGemmKernel = typename', 1)
s = s.replace('// Non-transposed B operand.\n  ::cutlass::bfloat16_t,\n  ::cutlass::layout::RowMajor,', '// B can be NN (row major) or NT (column major).\n  ::cutlass::bfloat16_t,\n  LayoutB,', 1)
s = s.replace('using GemmGroupedNN = ::cutlass::gemm::device::GemmGrouped<GroupedGemmKernelNN>;', 'using GemmGroupedNN = ::cutlass::gemm::device::GemmGrouped<GroupedGemmKernel<::cutlass::layout::RowMajor>>;\nusing GemmGroupedNT = ::cutlass::gemm::device::GemmGrouped<GroupedGemmKernel<::cutlass::layout::ColumnMajor>>;')
s = s.replace('MakeProblemSizes(torch::Tensor b, torch::Tensor batch_sizes)', 'MakeProblemSizes(torch::Tensor b, torch::Tensor batch_sizes, bool trans_b)')
s = s.replace('const size_t k = b.size(1), n = b.size(2);', 'const size_t k = b.size(trans_b ? 2 : 1), n = b.size(trans_b ? 1 : 2);', 1)
s = s.replace('torch::Tensor batch_sizes) {\n  auto problem_sizes_host = MakeProblemSizes(b, batch_sizes);', 'torch::Tensor batch_sizes, bool trans_b, std::vector<torch::Tensor>& storage) {\n  auto problem_sizes_host = MakeProblemSizes(b, batch_sizes, trans_b);', 1)
s = s.replace('  return arguments;\n}', '  storage = {lda, ldb, ldc, ptr_a, ptr_b, ptr_c, problem_sizes};\n  return arguments;\n}', 1)
s = s.replace('torch::Tensor CutlassGroupedGemm(', 'template <typename Gemm>\ntorch::Tensor CutlassGroupedGemm(', 1)
s = s.replace('  using Gemm = GemmGroupedNN;\n  Gemm gemm;', '  Gemm gemm;', 1)
s = s.replace('auto arguments = MakeArguments<Gemm>(a, b, c, batch_sizes);', 'std::vector<torch::Tensor> storage;\n  auto arguments = MakeArguments<Gemm>(a, b, c, batch_sizes,\n      std::is_same<typename Gemm::LayoutB, ::cutlass::layout::ColumnMajor>::value, storage);', 1)
start = s.index('  // NOTE: Use cuBLAS for SM90 until CUTLASS supports SM90-optimized grouped-gemm.')
end = s.index('\n#endif', start) + len('\n#endif')
s = s[:start] + '''  // Explicit SM80 tensor-core kernels also support SM89. Never select the
  // process-global cuBLAS fallback when decoder layers use different devices.
  if (trans_b) {
    CutlassGroupedGemm<GemmGroupedNT>(a, b, c, batch_sizes);
  } else {
    CutlassGroupedGemm<GemmGroupedNN>(a, b, c, batch_sizes);
  }
  return;''' + s[end:]
s = s.replace('#include "grouped_gemm.h"', '#include "grouped_gemm.h"\n#include <type_traits>', 1)
p.write_text(s)
p = root/'csrc/ops.cu'
s = subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:csrc/ops.cu'], text=True)
s = s.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {', 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.attr("qwen36_cutlass_only") = true;', 1)
p.write_text(s)
patch = subprocess.check_output(['git', '-C', str(root), 'diff', '--', 'csrc/grouped_gemm.cu', 'csrc/ops.cu'])
Path('reports/setup-qwen36-runtime/sm89-cutlass.patch').write_bytes(patch)
print(json.dumps({'grouped_gemm_revision':'172fada89fa7364fe5d026b3a0dfab58b591ffdd','cutlass_revision':'8783c41851cd3582490e04e69e0cd756a8c1db7f','patch_sha256':hashlib.sha256(patch).hexdigest(),'patch_bytes':len(patch)},indent=2))

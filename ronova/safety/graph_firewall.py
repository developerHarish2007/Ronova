"""
RONOVA Offline ONNX Graph Firewall:
Performs static-only structural analysis of ONNX model graphs before runtime loading or inference.
Enforces strict operator allowlisting, rejects unapproved or custom operator domains,
and validates input tensor dimensions against safe memory and element boundaries.
Never executes the model.
"""

from typing import Dict, Any, List, Optional, Set, Tuple, Union
from pathlib import Path
import onnx
from pydantic import BaseModel, Field


# Standard safe ONNX operators approved for offline machine learning inference
STANDARD_SAFE_OPERATORS: Set[str] = {
    "Abs",
    "Acos",
    "Acosh",
    "Add",
    "And",
    "ArgMax",
    "ArgMin",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "AveragePool",
    "BatchNormalization",
    "BitShift",
    "Cast",
    "CastLike",
    "Ceil",
    "Celu",
    "Clip",
    "Compress",
    "Concat",
    "ConcatFromSequence",
    "Constant",
    "ConstantOfShape",
    "Conv",
    "ConvInteger",
    "ConvTranspose",
    "Cos",
    "Cosh",
    "CumSum",
    "DepthToSpace",
    "DequantizeLinear",
    "Div",
    "Dropout",
    "Einsum",
    "Elu",
    "Equal",
    "Erf",
    "Exp",
    "Expand",
    "EyeLike",
    "Flatten",
    "Floor",
    "GRU",
    "Gather",
    "GatherElements",
    "GatherND",
    "Gelu",
    "Gemm",
    "GlobalAveragePool",
    "GlobalLpPool",
    "GlobalMaxPool",
    "Greater",
    "GreaterOrEqual",
    "GridSample",
    "HardSigmoid",
    "HardSwish",
    "Hardmax",
    "Identity",
    "InstanceNormalization",
    "IsInf",
    "IsNaN",
    "LRN",
    "LSTM",
    "LayerNormalization",
    "LeakyRelu",
    "Less",
    "LessOrEqual",
    "Log",
    "LogSoftmax",
    "LpNormalization",
    "LpPool",
    "MatMul",
    "MatMulInteger",
    "Max",
    "MaxPool",
    "MaxRoiPool",
    "MaxUnpool",
    "Mean",
    "MeanVarianceNormalization",
    "Min",
    "Mod",
    "Mul",
    "Multinomial",
    "Neg",
    "NonMaxSuppression",
    "NonZero",
    "Not",
    "OneHot",
    "Or",
    "PRelu",
    "Pad",
    "Pow",
    "QLinearConv",
    "QLinearMatMul",
    "QuantizeLinear",
    "RNN",
    "Range",
    "Reciprocal",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
    "Relu",
    "Reshape",
    "Resize",
    "ReverseSequence",
    "RoiAlign",
    "Round",
    "Scan",
    "Scatter",
    "ScatterElements",
    "ScatterND",
    "Selu",
    "SequenceAt",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceInsert",
    "SequenceLength",
    "Shape",
    "Shrink",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Size",
    "Slice",
    "Softmax",
    "SoftmaxCrossEntropyLoss",
    "Softplus",
    "Softsign",
    "SpaceToDepth",
    "Split",
    "SplitToSequence",
    "Sqrt",
    "Squeeze",
    "StringNormalizer",
    "Sub",
    "Sum",
    "Tan",
    "Tanh",
    "TfIdfVectorizer",
    "ThresholdedRelu",
    "Tile",
    "TopK",
    "Transpose",
    "Trilu",
    "Unique",
    "Unsqueeze",
    "Upsample",
    "Where",
    "Xor",
}

# Standard approved ONNX domains
APPROVED_DOMAINS: Set[str] = {"", "ai.onnx", "ai.onnx.ml"}


class GraphFirewallResult(BaseModel):
    passed: bool
    checked_operators: List[str] = Field(default_factory=list)
    disallowed_operators: List[str] = Field(default_factory=list)
    input_shapes: Dict[str, List[Union[int, str]]] = Field(default_factory=dict)
    total_input_elements: int = 0
    violations: List[str] = Field(default_factory=list)


class ONNXGraphFirewall:
    """
    Static ONNX Graph Firewall:
    Inspects model operators, operator domains, and input tensor geometries.
    Blocks execution if unapproved operators or boundary violations are detected.
    Never executes model code or initializes execution providers.
    """

    def __init__(
        self,
        allowed_operators: Optional[Set[str]] = None,
        max_input_elements: int = 16_000_000,
        max_dimension_size: int = 65_536,
        max_nodes: int = 50_000,
    ):
        self.allowed_operators = set(allowed_operators) if allowed_operators is not None else STANDARD_SAFE_OPERATORS
        self.max_input_elements = max_input_elements
        self.max_dimension_size = max_dimension_size
        self.max_nodes = max_nodes

    def _inspect_nodes_recursive(
        self,
        nodes: List[onnx.NodeProto],
        checked_ops: Set[str],
        disallowed_ops: Set[str],
        violations: List[str],
    ) -> int:
        """Recursively checks operator types and domains in graph nodes and subgraphs."""
        count = 0
        for node in nodes:
            count += 1
            op_type = getattr(node, "op_type", "")
            domain = getattr(node, "domain", "")

            checked_ops.add(op_type)

            # Check domain validity
            if domain not in APPROVED_DOMAINS:
                full_op = f"{domain}::{op_type}" if domain else op_type
                disallowed_ops.add(full_op)
                violations.append(
                    f"Disallowed operator domain '{domain}' for operator '{op_type}'"
                )

            # Check operator allowlist
            if op_type not in self.allowed_operators:
                disallowed_ops.add(op_type)
                violations.append(
                    f"Unapproved or custom operator '{op_type}' is not permitted by Graph Firewall policy"
                )

            # Check inner subgraphs (e.g. in If/Loop/Scan attribute bodies)
            for attr in getattr(node, "attribute", []):
                if attr.HasField("g"):
                    count += self._inspect_nodes_recursive(
                        attr.g.node, checked_ops, disallowed_ops, violations
                    )
                for sub_g in attr.graphs:
                    count += self._inspect_nodes_recursive(
                        sub_g.node, checked_ops, disallowed_ops, violations
                    )

        return count

    def inspect_graph(self, model: onnx.ModelProto) -> GraphFirewallResult:
        """
        Inspects an ONNX ModelProto structure statically.
        Returns a GraphFirewallResult with allowlist and dimension compliance details.
        """
        violations: List[str] = []
        checked_ops: Set[str] = set()
        disallowed_ops: Set[str] = set()
        input_shapes: Dict[str, List[Union[int, str]]] = {}
        total_input_elements = 0

        graph = getattr(model, "graph", None)
        if graph is None:
            return GraphFirewallResult(
                passed=False,
                checked_operators=[],
                disallowed_operators=[],
                input_shapes={},
                total_input_elements=0,
                violations=["Model does not contain a valid ONNX graph."],
            )

        # 1. Operator and Domain Inspection
        total_nodes = self._inspect_nodes_recursive(
            graph.node, checked_ops, disallowed_ops, violations
        )

        if total_nodes > self.max_nodes:
            violations.append(
                f"Graph node count ({total_nodes}) exceeds maximum allowed node limit ({self.max_nodes})"
            )

        # 2. Input Dimensions & Shape Inspection
        initializer_names = {init.name for init in graph.initializer}
        feed_inputs = [inp for inp in graph.input if inp.name not in initializer_names]

        for inp in feed_inputs:
            name = inp.name
            shape: List[Union[int, str]] = []
            tensor_elements = 1
            has_positive_dim = False

            if inp.type.HasField("tensor_type") and inp.type.tensor_type.HasField("shape"):
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        dim_val = dim.dim_value
                        shape.append(dim_val)
                        if dim_val > self.max_dimension_size:
                            violations.append(
                                f"Input '{name}' dimension ({dim_val}) exceeds maximum dimension size limit ({self.max_dimension_size})"
                            )
                        if dim_val > 0:
                            tensor_elements *= dim_val
                            has_positive_dim = True
                    elif dim.HasField("dim_param"):
                        param_name = dim.dim_param or "?"
                        shape.append(param_name)
                        # Assume symbolic/batch dimension = 1 for element budget estimation
                        tensor_elements *= 1
                    else:
                        shape.append("?")
                        tensor_elements *= 1

            input_shapes[name] = shape

            if has_positive_dim:
                if tensor_elements > self.max_input_elements:
                    violations.append(
                        f"Input '{name}' total element count ({tensor_elements:,}) exceeds maximum allowed input tensor limit ({self.max_input_elements:,})"
                    )
                total_input_elements += tensor_elements

        if total_input_elements > self.max_input_elements:
            violations.append(
                f"Combined input element count ({total_input_elements:,}) exceeds maximum allowed element budget ({self.max_input_elements:,})"
            )

        passed = len(violations) == 0

        return GraphFirewallResult(
            passed=passed,
            checked_operators=sorted(list(checked_ops)),
            disallowed_operators=sorted(list(disallowed_ops)),
            input_shapes=input_shapes,
            total_input_elements=total_input_elements,
            violations=violations,
        )

    def scan_file(self, file_path: Union[str, Path]) -> GraphFirewallResult:
        """
        Loads and statically scans an ONNX file without execution.
        """
        path = Path(file_path)
        if not path.exists():
            return GraphFirewallResult(
                passed=False,
                checked_operators=[],
                disallowed_operators=[],
                input_shapes={},
                total_input_elements=0,
                violations=[f"Model file not found: {file_path}"],
            )

        try:
            model = onnx.load(str(path))
            return self.inspect_graph(model)
        except Exception as e:
            return GraphFirewallResult(
                passed=False,
                checked_operators=[],
                disallowed_operators=[],
                input_shapes={},
                total_input_elements=0,
                violations=[f"ONNX static parsing failed: {str(e)}"],
            )

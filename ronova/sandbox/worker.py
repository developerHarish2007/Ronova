import sys
import json
import numpy as np
import onnxruntime as ort


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: python -m ronova.sandbox.worker <model_path> <input_npy_path> <output_npy_path> <metadata_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = sys.argv[1]
    input_npy_path = sys.argv[2]
    output_npy_path = sys.argv[3]
    metadata_json_path = sys.argv[4]

    try:
        # Load input array
        input_array = np.load(input_npy_path)
        if input_array.dtype != np.float32:
            input_array = input_array.astype(np.float32)

        # Set session options for strict CPU execution
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False

        session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        raw_outputs = session.run([output_name], {input_name: input_array})[0]

        # Compute softmax if raw outputs are logits
        exp_outputs = np.exp(raw_outputs - np.max(raw_outputs, axis=-1, keepdims=True))
        probs = exp_outputs / np.sum(exp_outputs, axis=-1, keepdims=True)

        np.save(output_npy_path, probs)

        metadata = {
            "onnx_input_name": input_name,
            "onnx_output_name": output_name,
            "input_shape": list(input_array.shape),
            "output_shape": list(probs.shape),
            "providers_used": session.get_providers(),
        }
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        sys.exit(0)
    except Exception as e:
        print(f"WORKER_ERROR: {str(e)}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
